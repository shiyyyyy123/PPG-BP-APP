# configs.py - 移动部署优化版
import torch
import os

class Config:
    # 数据参数
    input_dim = None  # 将在数据加载时动态设置
    output_dim = 2    # 收缩压和舒张压两个输出

    max_features = 80
    MAX_FEATURES = 100
    SP_power = 0.75
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练参数 - 优化后
    batch_size = 32  # 降低批量大小，增加更新频率
    epochs = 100     # 增加最大训练轮次
    patience = 40   # 增加早停耐心值
    max_lr = 1e-4    # 降低学习率，使学习更稳定
    weight_decay = 1e-3  # 增加权重衰减，减少过拟合

    # 损失函数参数
    huber_delta = 1.5        # Huber损失的delta参数
    sp_weight = 0.7          # 收缩压损失权重（增加）
    dp_weight = 0.3          # 舒张压损失权重

    # 蒸馏参数 - 优化后
    distill_alpha = 0.3      # 增加监督学习的权重
    distill_temp = 2.0       # 降低温度参数
    feature_loss_weight = 1.0  # 增加特征蒸馏的权重
    soft_loss_weight = 0.0    # 相应减少软目标权重

    # ==================== 多层特征蒸馏配置 ====================
    # 控制启用哪些层的特征蒸馏
    # 建议策略：
    # 1. 只用最终层 (原始方法): use_layer1=False, use_layer2=False, use_layer3=False, use_final=True
    # 2. 深层特征 (推荐): use_layer1=False, use_layer2=True, use_layer3=True, use_final=True
    # 3. 全部特征 (实验): 全部设为True
    
    distill_use_layer1 = True  # 是否蒸馏layer1 (浅层，不推荐，易引入噪声)
    distill_use_layer2 = True  # 是否蒸馏layer2 (中层，可选)
    distill_use_layer3 = True   # 是否蒸馏layer3 (深层，推荐)
    distill_use_final = True    # 是否蒸馏最终层 (强烈推荐)
    
    # 各层特征蒸馏的权重 (归一化后使用)
    # 注意：权重越大，该层对损失的贡献越大
    layer1_weight = 0.3   # 浅层权重 (如果启用)
    layer2_weight = 0.5   # 中层权重 (如果启用)
    layer3_weight = 0.7   # 深层权重 (如果启用)
    final_weight = 1.0    # 最终层权重 (最重要)
    
    # 旧的配置（保留用于兼容）
    feature_layers = {
        'early': {
            'teacher_layer': 'layer1',  # ResNet的第一层
            'student_layer': 'bneck1',  # MobileNetV3的第一个bneck
            'weight': 0.2
        },
        'middle': {
            'teacher_layer': 'layer2',  # ResNet的第二层
            'student_layer': 'bneck4',  # MobileNetV3的中间bneck
            'weight': 0.3
        },
        'late': {
            'teacher_layer': 'layer3',  # ResNet的第三层
            'student_layer': 'bneck7',  # MobileNetV3的后期bneck
            'weight': 0.5
        }
    }

    # 预热参数
    pct_start = 0.3
    
    # 移动优化参数 - 增强版
    quantize = True         # 是否启用量化
    prune_threshold = 0.015 # 增加剪枝阈值
    quantize_dtype = 'qint8'# 量化数据类型
    
    # 模型保存路径
    model_dir = "models"
    teacher_path = os.path.join(model_dir, "teacher_best.pth")
    student_path = os.path.join(model_dir, "student_best.pth")
    onnx_path = os.path.join(model_dir, "student_mobile.onnx")
    
    def __init__(self):
        """确保模型目录存在"""
        os.makedirs(self.model_dir, exist_ok=True)
