# train_distill.py (适配MobileNetV3模型)
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.teacher_model import BloodPressureTeacher
from models.student_model import MobileBPStudent
from configs import Config
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import os
import matplotlib.pyplot as plt
import matplotlib
# 解决中文字体显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置matplotlib后端，避免图像尺寸问题
matplotlib.use('Agg')  # 使用非交互式后端


class EnhancedDistillLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.distill_alpha
        self.temp = config.distill_temp
        self.delta = config.huber_delta
        self.feature_loss_weight = config.feature_loss_weight
        self.soft_loss_weight = config.soft_loss_weight
        self.sp_weight = config.sp_weight
        self.dp_weight = config.dp_weight
        
        # ========== 多层蒸馏配置 ==========
        # 可以通过config控制启用哪些层

        self.use_layer1 = getattr(config, 'distill_use_layer1', False)  # 默认关闭浅层
        self.use_layer2 = getattr(config, 'distill_use_layer2', True)   # 默认开启中层
        self.use_layer3 = getattr(config, 'distill_use_layer3', True)   # 默认开启深层
        self.use_final = getattr(config, 'distill_use_final', True)     # 默认开启最终层
        
        # 特征权重配置（归一化处理，避免权重过大）
        # 建议策略：深层权重递增，但总和控制在合理范围
        self.layer1_weight = getattr(config, 'layer1_weight', 0.3)
        self.layer2_weight = getattr(config, 'layer2_weight', 0.5)
        self.layer3_weight = getattr(config, 'layer3_weight', 0.7)
        self.final_weight = getattr(config, 'final_weight', 1.0)
        
        # 学生模型实际特征维度 (根据student_model.py的cfg配置):
        # student_feats[0]: 16维 (初始特征)
        # student_feats[1]: 40维 (block3输出)
        # student_feats[2]: 48维 (block7输出)  
        # student_feats[3]: 96维 (block10输出)
        # student_feats[4]: 768维 (conv层输出)
        # student_feats[5]: 1024维 (feature_adapter输出)
        
        # 教师模型特征维度:
        # teacher_layer1: 64维
        # teacher_layer2: 128维
        # teacher_layer3: 256维
        
        # 适配器1: 学生早期特征 -> 教师layer1 (64维)
        if self.use_layer1:
            self.adapter_layer1 = nn.Linear(40, 64)
            nn.init.xavier_uniform_(self.adapter_layer1.weight)
            nn.init.zeros_(self.adapter_layer1.bias)
        
        # 适配器2: 学生中期特征 -> 教师layer2 (128维)
        if self.use_layer2:
            self.adapter_layer2 = nn.Linear(48, 128)
            nn.init.xavier_uniform_(self.adapter_layer2.weight)
            nn.init.zeros_(self.adapter_layer2.bias)
        
        # 适配器3: 学生后期特征 -> 教师layer3 (256维)
        if self.use_layer3:
            self.adapter_layer3 = nn.Linear(96, 256)
            nn.init.xavier_uniform_(self.adapter_layer3.weight)
            nn.init.zeros_(self.adapter_layer3.bias)
        
        # 额外的最终特征适配器（保留原有的深层蒸馏）
        if self.use_final:
            self.adapter_final = nn.Linear(1024, 256)
            nn.init.xavier_uniform_(self.adapter_final.weight)
            nn.init.zeros_(self.adapter_final.bias)
            
            # 添加收缩压特定的注意力机制（仅用于最终层）
            self.sp_attention = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.Sigmoid()
            )
        
        # 打印蒸馏配置
        print("\n=== 多层特征蒸馏配置 ===")
        print(f"Layer1 (浅层): {'启用' if self.use_layer1 else '禁用'} (权重={self.layer1_weight})")
        print(f"Layer2 (中层): {'启用' if self.use_layer2 else '禁用'} (权重={self.layer2_weight})")
        print(f"Layer3 (深层): {'启用' if self.use_layer3 else '禁用'} (权重={self.layer3_weight})")
        print(f"Final  (最终): {'启用' if self.use_final else '禁用'} (权重={self.final_weight})")
        print("========================\n")

    def huber_loss(self, pred, target):
        """改进的Huber损失函数，对收缩压和舒张压分别加权"""
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        
        # 分别计算收缩压(索引0)和舒张压(索引1)的损失
        sp_loss = 0.5 * quadratic[:, 0] ** 2 + self.delta * (abs_error[:, 0] - quadratic[:, 0])
        dp_loss = 0.5 * quadratic[:, 1] ** 2 + self.delta * (abs_error[:, 1] - quadratic[:, 1])
        
        # 使用配置的权重
        return self.sp_weight * sp_loss + self.dp_weight * dp_loss

    def distributional_regression_loss(self, student_pred, teacher_pred):
        """
        分布式回归蒸馏损失 - 正确的回归任务蒸馏方法
        将预测值转换为高斯分布，然后计算KL散度
        
        Args:
            student_pred: 学生模型预测 [batch_size, 2] (SP, DP)
            teacher_pred: 教师模型预测 [batch_size, 2] (SP, DP)  
        Returns:
            kl_loss: 分布间的KL散度损失
        """
        try:
            # 使用温度参数作为分布的标准差，表示预测的不确定性
            # 温度越大，表示对预测越不确定
            teacher_dist = torch.distributions.Normal(teacher_pred, self.temp)
            student_dist = torch.distributions.Normal(student_pred, self.temp)
            
            # 计算KL散度：衡量两个概率分布的相似性
            # KL(P_teacher || P_student) = E_P[log(P_teacher/P_student)]
            kl_div = torch.distributions.kl_divergence(teacher_dist, student_dist)
            
            # 对所有维度求平均
            return kl_div.mean()
            
        except Exception as e:
            print(f"分布式回归蒸馏计算失败: {e}")
            return torch.tensor(0.0, device=student_pred.device, requires_grad=True)

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        改进的蒸馏损失函数，使用分布式回归蒸馏和多层特征蒸馏
        - student_outputs: (预测结果, 特征列表)
        - teacher_outputs: (预测结果, layer1特征, layer2特征, layer3特征)
        - targets: 真实标签
        
        损失组成:
        1. 监督学习损失: 加权Huber损失，直接监督预测准确性
        2. 分布式蒸馏损失: 通过高斯分布建模预测不确定性，传递教师的"软知识"
        3. 多层特征蒸馏损失: 对齐3层中间特征表示，传递表征学习能力
        """
        try:
            student_pred, student_feats = student_outputs
            teacher_pred, teacher_layer1, teacher_layer2, teacher_layer3 = teacher_outputs

            # ============ 多层特征蒸馏 ============
            # 学生模型特征对应关系:
            # student_feats[0]: 初始特征 (16维)
            # student_feats[1]: block3 特征 (40维) -> 对齐教师layer1
            # student_feats[2]: block7 特征 (48维) -> 对齐教师layer2
            # student_feats[3]: block10特征 (96维) -> 对齐教师layer3
            # student_feats[4]: 最终特征 (768维)
            # student_feats[5]: 适配特征 (1024维) -> 额外对齐教师layer3
            
            feature_losses = []
            feature_weights = []
            
            # Layer 1 蒸馏 (浅层特征) - 可选
            if self.use_layer1 and len(student_feats) > 1:
                try:
                    student_feat_1 = student_feats[1]  # block3输出
                    adapted_feat_1 = self.adapter_layer1(student_feat_1)
                    loss_1 = F.mse_loss(adapted_feat_1, teacher_layer1)
                    feature_losses.append(loss_1)
                    feature_weights.append(self.layer1_weight)
                except Exception as e:
                    print(f"Layer1蒸馏失败: {e}, student_feat shape: {student_feats[1].shape if len(student_feats) > 1 else 'N/A'}")
            
            # Layer 2 蒸馏 (中层特征) - 可选
            if self.use_layer2 and len(student_feats) > 2:
                try:
                    student_feat_2 = student_feats[2]  # block7输出
                    adapted_feat_2 = self.adapter_layer2(student_feat_2)
                    loss_2 = F.mse_loss(adapted_feat_2, teacher_layer2)
                    feature_losses.append(loss_2)
                    feature_weights.append(self.layer2_weight)
                except Exception as e:
                    print(f"Layer2蒸馏失败: {e}, student_feat shape: {student_feats[2].shape if len(student_feats) > 2 else 'N/A'}")
            
            # Layer 3 蒸馏 (深层特征) - 可选
            if self.use_layer3 and len(student_feats) > 3:
                try:
                    student_feat_3 = student_feats[3]  # block10输出
                    adapted_feat_3 = self.adapter_layer3(student_feat_3)
                    loss_3 = F.mse_loss(adapted_feat_3, teacher_layer3)
                    feature_losses.append(loss_3)
                    feature_weights.append(self.layer3_weight)
                except Exception as e:
                    print(f"Layer3蒸馏失败: {e}, student_feat shape: {student_feats[3].shape if len(student_feats) > 3 else 'N/A'}")
            
            # 最终特征蒸馏（深层特征，带注意力机制） - 可选
            if self.use_final and len(student_feats) > 5:
                try:
                    final_student_feat = student_feats[5]  # 1024维适配特征
                    adapted_final_feat = self.adapter_final(final_student_feat)
                    
                    # 应用收缩压特定的注意力权重
                    attention_weights = self.sp_attention(adapted_final_feat)
                    attended_feature = adapted_final_feat * attention_weights
                    
                    final_loss = F.mse_loss(attended_feature, teacher_layer3)
                    feature_losses.append(final_loss)
                    feature_weights.append(self.final_weight)
                except Exception as e:
                    print(f"最终特征蒸馏失败: {e}")
            
            # 聚合特征损失（加权平均，归一化）
            if len(feature_losses) > 0:
                # 使用配置的权重，并归一化
                total_weight = sum(feature_weights)
                feature_loss = sum(w * loss for w, loss in zip(feature_weights, feature_losses)) / total_weight
            else:
                feature_loss = torch.tensor(0.0, device=student_pred.device, requires_grad=True)
                print("警告：没有计算任何特征蒸馏损失")
            
            # ============ 分布式回归蒸馏损失 ============
            soft_loss = self.distributional_regression_loss(student_pred, teacher_pred.detach())
            
            # ============ 监督学习损失 ============
            target_loss = self.huber_loss(student_pred, targets).mean()
            
            # ============ 总损失 ============
            total_loss = (
                self.alpha * target_loss + 
                self.soft_loss_weight * (1 - self.alpha) * soft_loss +
                self.feature_loss_weight * (1 - self.alpha) * feature_loss
            )
            
            return total_loss
            
        except Exception as e:
            print(f"蒸馏损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=targets.device, requires_grad=True)


def train_distill():
    """改进的知识蒸馏训练函数"""
    config = Config()

    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)

    # 生成多层蒸馏配置的文件名标识
    layer_config_str = ""
    layers_enabled = []
    if getattr(config, 'distill_use_layer1', False):
        layers_enabled.append("L1")
    if getattr(config, 'distill_use_layer2', False):
        layers_enabled.append("L2")
    if getattr(config, 'distill_use_layer3', False):
        layers_enabled.append("L3")
    if getattr(config, 'distill_use_final', True):
        layers_enabled.append("Final")
    
    if len(layers_enabled) > 0:
        layer_config_str = f"-layers={'_'.join(layers_enabled)}"
    else:
        layer_config_str = "-layers=None"
    
    print(f"\n📁 模型文件名标识: {layer_config_str}")

    # 记录训练和验证损失
    train_losses = []  # 蒸馏损失（用于优化）
    val_losses = []    # 验证MAE损失
    train_mae_losses = []  # 训练MAE损失（用于对比）

    # 加载数据（添加allow_pickle=True）
    X_train = np.load("data/processed/X_train.npy")
    selected_features = np.load("data/processed/selected_features.npy", allow_pickle=True)
    config.input_dim = X_train.shape[1]
    print(f"输入特征维度: {config.input_dim}")

    # 初始化教师模型
    teacher = BloodPressureTeacher(config).to(config.device)
    try:
        teacher.load_state_dict(torch.load("models/teacher_aug_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", map_location=config.device))
        print("成功加载教师模型权重")
    except Exception as e:
        print(f"加载教师模型失败: {e}")
        print("请先训练教师模型！")
        return
    
    teacher.eval()

    # 生成教师指导信号 - 提取所有3层特征
    with torch.no_grad():
        print("正在生成教师模型的所有中间层特征...")
        X_tensor = torch.FloatTensor(X_train).to(config.device)
        
        # 分批处理以避免内存溢出
        batch_size = 1024
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        all_teacher_preds = []
        all_teacher_layer1 = []
        all_teacher_layer2 = []
        all_teacher_layer3 = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            batch_X = X_tensor[start_idx:end_idx]
            
            # 教师模型在训练模式下会返回特征
            teacher.train()  # 临时切换到训练模式以获取特征
            outputs = teacher(batch_X)
            teacher.eval()   # 切换回评估模式
            
            if isinstance(outputs, tuple):
                pred, layer_outputs = outputs
                
                # 提取所有3层特征
                layer1_feat = layer_outputs.get('layer1', None)
                layer2_feat = layer_outputs.get('layer2', None)
                layer3_feat = layer_outputs.get('layer3', None)
                
                # 检查特征是否存在
                if layer1_feat is None or layer2_feat is None or layer3_feat is None:
                    print("警告：部分中间层特征缺失")
                    print(f"layer1: {layer1_feat is not None}, layer2: {layer2_feat is not None}, layer3: {layer3_feat is not None}")
            else:
                pred = outputs
                layer1_feat = layer2_feat = layer3_feat = None
                
            all_teacher_preds.append(pred.cpu())
            
            # 处理每一层特征（如果是4D张量则池化）
            if layer1_feat is not None:
                if len(layer1_feat.shape) == 4:
                    layer1_feat = torch.mean(layer1_feat, dim=[2, 3])  # [B, C]
                all_teacher_layer1.append(layer1_feat.cpu())
            
            if layer2_feat is not None:
                if len(layer2_feat.shape) == 4:
                    layer2_feat = torch.mean(layer2_feat, dim=[2, 3])  # [B, C]
                all_teacher_layer2.append(layer2_feat.cpu())
            
            if layer3_feat is not None:
                if len(layer3_feat.shape) == 4:
                    layer3_feat = torch.mean(layer3_feat, dim=[2, 3])  # [B, C]
                all_teacher_layer3.append(layer3_feat.cpu())
        
        # 合并所有批次
        teacher_pred = torch.cat(all_teacher_preds, dim=0)
        
        # 组装教师特征
        if len(all_teacher_layer1) > 0 and len(all_teacher_layer2) > 0 and len(all_teacher_layer3) > 0:
            teacher_layer1 = torch.cat(all_teacher_layer1, dim=0)
            teacher_layer2 = torch.cat(all_teacher_layer2, dim=0)
            teacher_layer3 = torch.cat(all_teacher_layer3, dim=0)
            teacher_outputs = (teacher_pred, teacher_layer1, teacher_layer2, teacher_layer3)
            print(f"教师模型特征生成完毕:")
            print(f"  layer1形状: {teacher_layer1.shape}")
            print(f"  layer2形状: {teacher_layer2.shape}")
            print(f"  layer3形状: {teacher_layer3.shape}")
        else:
            print("错误：教师模型特征提取失败，无法进行知识蒸馏")
            return

    # 初始化学生模型
    student = MobileBPStudent(config).to(config.device)
    print("创建MobileNetV3学生模型")
    
    # 预热阶段 - 简单训练以稳定模型
    print("开始学生模型预热阶段...")
    try:
        X_warmup = torch.FloatTensor(X_train[:1000]).to(config.device)
        y_warmup = torch.FloatTensor(np.load("data/processed/y_train.npy")[:1000]).to(config.device)
        
        # 简单的MSE损失
        warmup_criterion = nn.MSELoss().to(config.device)
        warmup_optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        student.train()
        # 3轮预热训练
        for i in range(3):
            warmup_optimizer.zero_grad()
            pred, _ = student(X_warmup)
            loss = warmup_criterion(pred, y_warmup)
            loss.backward()
            warmup_optimizer.step()
            print(f"预热轮次 {i+1}/3, 损失: {loss.item():.4f}")
            
        print("预热阶段完成")
    except Exception as e:
        print(f"预热阶段失败: {e}，但继续训练")

    # 数据集 - 包含3层教师特征
    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(np.load("data/processed/y_train.npy")),
        teacher_outputs[0],  # 教师预测
        teacher_outputs[1],  # 教师layer1特征
        teacher_outputs[2],  # 教师layer2特征
        teacher_outputs[3]   # 教师layer3特征
    )

    # 数据加载器
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    # 优化器设置 - 使用差异化权重衰减
    param_groups = [
        {'params': [], 'weight_decay': config.weight_decay * 1.5},  # 特征提取层（较大权重衰减）
        {'params': [], 'weight_decay': config.weight_decay}         # 预测层（标准权重衰减）
    ]
    
    # 将参数分配到不同组
    for name, param in student.named_parameters():
        if 'classifier' in name or 'fc' in name or 'head' in name:
            param_groups[1]['params'].append(param)  # 预测层
        else:
            param_groups[0]['params'].append(param)  # 特征提取层
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.max_lr,
        eps=1e-6  # 提高数值稳定性
    )

    # 改进的学习率调度 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr * 1.2,  # 稍微提高峰值学习率
        total_steps=config.epochs * len(loader),
        pct_start=0.25,  # 预热阶段比例
        div_factor=20,  # 起始学习率 = max_lr/20
        final_div_factor=20000,  # 最终学习率 = max_lr/20000
        anneal_strategy='cos'  # 使用余弦退火
    )

    # 改进EMA - 使用动态衰减率
    ema = ExponentialMovingAverage(student.parameters(), decay=0.996)

    # 损失函数
    criterion = EnhancedDistillLoss(config).to(config.device)

    best_mae = float('inf')
    patience_counter = 0
    
    # 测试一个批次的前向传播，确保模型能正常工作
    print("正在测试学生模型前向传播...")
    student.eval()
    with torch.no_grad():
        try:
            test_batch, _, test_teacher_pred, test_teacher_layer1, test_teacher_layer2, test_teacher_layer3 = next(iter(loader))
            test_batch = test_batch.to(config.device)
            print(f"测试批次形状: {test_batch.shape}")
            student_outputs, student_features = student(test_batch)
            print(f"学生模型输出形状: {student_outputs.shape}")
            print(f"学生模型特征数量: {len(student_features)}")
            print(f"学生模型特征形状: [", end="")
            for i, feat in enumerate(student_features):
                print(f"{feat.shape}", end=", " if i < len(student_features) - 1 else "]\n")
                
            # 测试特征适配是否正常工作
            print("测试多层特征适配器:")
            print(f"  教师layer1形状: {test_teacher_layer1.shape}")
            print(f"  教师layer2形状: {test_teacher_layer2.shape}")
            print(f"  教师layer3形状: {test_teacher_layer3.shape}")
            
            # 只测试已启用的适配器
            if criterion.use_layer1 and len(student_features) > 1:
                test_adapted_1 = criterion.adapter_layer1(student_features[1].to(config.device))
                print(f"  ✓ Layer1适配: 学生特征1 {student_features[1].shape} -> 适配后 {test_adapted_1.shape}")
            elif not criterion.use_layer1:
                print(f"  - Layer1适配: 未启用（跳过）")
                
            if criterion.use_layer2 and len(student_features) > 2:
                test_adapted_2 = criterion.adapter_layer2(student_features[2].to(config.device))
                print(f"  ✓ Layer2适配: 学生特征2 {student_features[2].shape} -> 适配后 {test_adapted_2.shape}")
            elif not criterion.use_layer2:
                print(f"  - Layer2适配: 未启用（跳过）")
                
            if criterion.use_layer3 and len(student_features) > 3:
                test_adapted_3 = criterion.adapter_layer3(student_features[3].to(config.device))
                print(f"  ✓ Layer3适配: 学生特征3 {student_features[3].shape} -> 适配后 {test_adapted_3.shape}")
            elif not criterion.use_layer3:
                print(f"  - Layer3适配: 未启用（跳过）")
                
            if criterion.use_final and len(student_features) > 5:
                test_adapted_final = criterion.adapter_final(student_features[5].to(config.device))
                print(f"  ✓ Final适配: 学生特征5 {student_features[5].shape} -> 适配后 {test_adapted_final.shape}")
            elif not criterion.use_final:
                print(f"  - Final适配: 未启用（跳过）")
            
            print("前向传播测试成功！")
        except Exception as e:
            print(f"前向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            print("尝试继续训练...")
    
    for epoch in range(config.epochs):
        student.train()
        total_loss = 0.0
        total_mae = 0.0  # 记录训练时的MAE
        progress_bar = tqdm(loader, desc=f"蒸馏 Epoch {epoch + 1}/{config.epochs}")

        for batch_idx, (X, y_true, y_teacher_pred, y_teacher_layer1, y_teacher_layer2, y_teacher_layer3) in enumerate(progress_bar):
            X = X.to(config.device)
            y_true = y_true.to(config.device)
            y_teacher = (
                y_teacher_pred.to(config.device), 
                y_teacher_layer1.to(config.device),
                y_teacher_layer2.to(config.device),
                y_teacher_layer3.to(config.device)
            )
            
            # 第一个批次打印形状信息
            if batch_idx == 0 and epoch == 0:
                print(f"\n批次数据形状: X={X.shape}, y_true={y_true.shape}")
                print(f"教师预测形状: {y_teacher_pred.shape}")
                print(f"教师layer1形状: {y_teacher_layer1.shape}")
                print(f"教师layer2形状: {y_teacher_layer2.shape}")
                print(f"教师layer3形状: {y_teacher_layer3.shape}")

            optimizer.zero_grad()
            
            # 使用try-except捕获可能的错误
            try:
                student_out = student(X)  # 返回 (预测, 特征列表)
                student_pred, _ = student_out
                
                # 计算蒸馏损失用于优化
                loss = criterion(student_out, y_teacher, y_true)
                
                # 计算纯MAE损失用于对比（与验证阶段一致）
                mae_loss = F.l1_loss(student_pred, y_true)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                ema.update()
                scheduler.step()
                
                total_loss += loss.item() * X.size(0)
                total_mae += mae_loss.item() * X.size(0)  # 累积MAE
                progress_bar.set_postfix({
                    'distill_loss': loss.item(), 
                    'mae': mae_loss.item()
                })
                
            except Exception as e:
                print(f"\n训练批次 {batch_idx} 发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 如果是第一个批次出错，打印更多调试信息
                if batch_idx == 0:
                    print(f"输入X统计: min={X.min().item()}, max={X.max().item()}, mean={X.mean().item()}")
                    # 尝试分析原因
                    try:
                        outputs, feats = student(X)
                        print(f"学生输出形状: {outputs.shape}")
                        print(f"学生特征数量: {len(feats)}")
                        for i, feat in enumerate(feats):
                            print(f"  学生特征{i}形状: {feat.shape}")
                        print(f"教师layer1形状: {y_teacher[1].shape}")
                        print(f"教师layer2形状: {y_teacher[2].shape}")
                        print(f"教师layer3形状: {y_teacher[3].shape}")
                    except Exception as inner_e:
                        print(f"调试分析失败: {inner_e}")
                if epoch == 0 and batch_idx == 0:
                    # 输出错误信息，但不中止训练
                    print("在第一个批次出现错误，但尝试继续训练")
                # 跳过这个批次
                continue

        # 验证阶段
        student.eval()
        X_val = torch.FloatTensor(np.load("data/processed/X_val.npy")).to(config.device)
        y_val = torch.FloatTensor(np.load("data/processed/y_val.npy")).to(config.device)

        with torch.no_grad():
            try:
                preds, _ = student(X_val)
                
                # 计算总体MAE
                val_mae = F.l1_loss(preds, y_val).item()
                
                # 单独计算收缩压(SP)和舒张压(DP)的MAE
                sp_mae = F.l1_loss(preds[:, 0], y_val[:, 0]).item()
                dp_mae = F.l1_loss(preds[:, 1], y_val[:, 1]).item()
                
                avg_loss = total_loss / len(loader.dataset)  # 平均蒸馏损失
                avg_mae = total_mae / len(loader.dataset)    # 平均训练MAE
                val_loss = val_mae  # 使用MAE作为验证损失

                # 记录损失
                train_losses.append(avg_loss)        # 蒸馏损失（用于优化）
                train_mae_losses.append(avg_mae)     # 训练MAE（用于对比）
                val_losses.append(val_loss)          # 验证MAE
                
                print(f"\nEpoch {epoch + 1} 摘要:")
                print(f"  蒸馏损失: {avg_loss:.4f} (优化目标)")
                print(f"  训练MAE: {avg_mae:.2f} mmHg (对比指标)")
                print(f"  验证MAE: {val_mae:.2f} mmHg")
                print(f"  收缩压MAE: {sp_mae:.2f} mmHg | 舒张压MAE: {dp_mae:.2f} mmHg")
                
                # 计算收缩压和舒张压的误差比例
                sp_dp_ratio = sp_mae / dp_mae if dp_mae > 0 else float('inf')
                print(f"  收缩压/舒张压误差比: {sp_dp_ratio:.2f}x")
                
                if val_mae < best_mae:
                    best_mae = val_mae
                    patience_counter = 0
                    with ema.average_parameters():
                        # 保存完整模型 - 包含layer配置信息
                        model_name = f"models/student_best{layer_config_str}-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.pth"
                        torch.save(student.state_dict(), model_name)
                        print(f"==> 保存PyTorch模型: {model_name}")
                        
                        # 导出到ONNX格式(用于移动部署) - 包含layer配置信息
                        try:
                            dummy_input = torch.randn(1, config.input_dim, device=config.device)
                            onnx_name = f"models/student_mobile{layer_config_str}-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.onnx"
                            torch.onnx.export(
                                student, 
                                dummy_input, 
                                onnx_name,
                                export_params=True,
                                opset_version=12,
                                input_names=['input'],
                                output_names=['output'],
                                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                            )
                            print(f"==> 保存ONNX模型: {onnx_name}")
                            print("==> ONNX模型可用于安卓部署！")
                        except Exception as e:
                            print(f"ONNX导出失败: {e}")
                    print("==> 保存EMA参数的最佳学生模型！")
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"早停: {config.patience}个epoch没有改进")
                        break

                # 绘制训练曲线 - 参考teacher模型，只更新一张图（避免生成过多文件）
                try:
                    # 主要MAE对比图（类似teacher模型的单一更新方式）
                    plt.figure(figsize=(10, 6))
                    epochs_range = range(1, len(train_mae_losses) + 1)
                    plt.plot(epochs_range, train_mae_losses, 'b-', label='Train MAE', linewidth=2, marker='o', markersize=2)
                    plt.plot(epochs_range, val_losses, 'r-', label='Val MAE', linewidth=2, marker='s', markersize=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('MAE (mmHg)')
                    plt.title('Knowledge Distillation Training - MAE Comparison')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 添加当前最佳MAE标注
                    if len(val_losses) > 0:
                        try:
                            best_epoch = int(np.argmin(val_losses))
                            best_val_mae = float(val_losses[best_epoch])
                            
                            # 简化的标注，避免坐标计算问题
                            plt.text(0.02, 0.98, f'Best Val MAE: {best_val_mae:.2f} mmHg (Epoch {best_epoch+1})',
                                   transform=plt.gca().transAxes, fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                                   verticalalignment='top')
                        except Exception as e:
                            print(f"标注添加失败: {e}")
                    
                    # 固定文件名，每次覆盖更新（类似teacher模型）- 包含layer配置信息
                    mae_fig_name = f'models/distill_training_mae{layer_config_str}-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.png'
                    plt.savefig(mae_fig_name, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # 蒸馏损失图（可选，用于监控优化过程）- 包含layer配置信息
                    plt.figure(figsize=(10, 5))
                    plt.plot(epochs_range, train_losses, 'g-', label='Distillation Loss', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Distillation Loss')
                    plt.title('Knowledge Distillation Training - Optimization Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 固定文件名，每次覆盖更新
                    loss_fig_name = f'models/distill_training_loss{layer_config_str}-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.png'
                    plt.savefig(loss_fig_name, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as plot_error:
                    print(f"绘图过程发生错误: {plot_error}")
                    print("跳过绘图，继续训练...")
                    plt.close('all')  # 关闭所有可能打开的图表
                        
            except Exception as e:
                print(f"验证阶段发生错误: {e}")
                # 跳过这个epoch的验证，但仍要记录空值以保持列表长度一致
                train_mae_losses.append(float('nan'))
                continue


if __name__ == "__main__":
    train_distill()
