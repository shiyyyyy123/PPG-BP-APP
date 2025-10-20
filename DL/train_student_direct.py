# train_student_direct.py - 直接训练学生模型（无知识蒸馏）
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.student_model import MobileBPStudent
from configs import Config
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import os
import time
import matplotlib.pyplot as plt
import matplotlib
# 解决中文字体显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置matplotlib后端，避免图像尺寸问题
matplotlib.use('Agg')  # 使用非交互式后端


class DirectLoss(nn.Module):
    """直接监督学习的损失函数，与教师模型保持一致"""
    def __init__(self, config):
        super().__init__()
        self.delta = config.huber_delta
        self.sp_weight = config.sp_weight
    
    def huber_loss(self, pred, target):
        """与教师模型相同的Huber损失实现"""
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear

    def forward(self, pred, target):
        """与教师模型相同的损失计算方式"""
        sp_loss = self.huber_loss(pred[:, 0], target[:, 0]).mean()
        dp_loss = self.huber_loss(pred[:, 1], target[:, 1]).mean()
        return self.sp_weight * sp_loss + (1 - self.sp_weight) * dp_loss


def train_student_direct():
    """直接训练学生模型（无知识蒸馏）"""
    print("开始直接训练学生模型，无知识蒸馏...")
    config = Config()

    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)

    # 加载数据
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    selected_features = np.load("data/processed/selected_features.npy", allow_pickle=True)
    
    config.input_dim = X_train.shape[1]
    print(f"输入特征维度: {config.input_dim}")

    # 初始化学生模型
    student = MobileBPStudent(config).to(config.device)
    print("创建MobileNetV3学生模型 (直接训练)")
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size*2,
        pin_memory=True
    )
    
    # 定义损失函数和优化器 - 与教师模型保持一致
    criterion = DirectLoss(config)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.max_lr,
        total_steps=config.epochs * len(train_loader),
        pct_start=config.pct_start
    )
    
    # EMA平滑模型权重 - 与教师模型保持一致
    ema = ExponentialMovingAverage(student.parameters(), decay=0.999)
    
    # 训练循环
    best_mae = float('inf')
    patience_counter = 0
    train_losses = []  # Huber损失（优化目标）
    val_losses = []    # Huber损失（验证）
    train_mae_losses = []  # 训练MAE（对比指标）
    val_mae_losses = []    # 验证MAE（对比指标）
    student_path = f"models/student_direct_best-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.pth"
    
    print(f"开始训练，最大轮次: {config.epochs}")
    try:
        for epoch in range(config.epochs):
            # 训练阶段
            student.train()
            train_loss = 0.0
            
            train_mae = 0.0  # 记录训练MAE
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(config.device), batch_y.to(config.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs, _ = student(batch_X)
                
                # 计算损失
                loss = criterion(outputs, batch_y)
                mae_loss = F.l1_loss(outputs, batch_y)  # 计算MAE用于对比
                
                # 反向传播和优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)  # 与教师模型一致
                optimizer.step()
                scheduler.step()
                ema.update()
                
                train_loss += loss.item() * batch_X.size(0)
                train_mae += mae_loss.item() * batch_X.size(0)
                progress_bar.set_postfix({
                    "huber_loss": f"{loss.item():.4f}", 
                    "mae": f"{mae_loss.item():.2f}"
                })
            
            train_loss = train_loss / len(train_loader.dataset)
            train_mae = train_mae / len(train_loader.dataset)
            train_losses.append(train_loss)
            train_mae_losses.append(train_mae)
            
            # 验证阶段
            student.eval()
            val_loss = 0.0
            val_mae_sp = 0.0
            val_mae_dp = 0.0
            
            with torch.no_grad(), ema.average_parameters():  # 使用EMA参数进行评估
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.device), batch_y.to(config.device)
                    outputs, _ = student(batch_X)
                    
                    val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
                    val_mae_sp += F.l1_loss(outputs[:, 0], batch_y[:, 0]).item() * batch_X.size(0)
                    val_mae_dp += F.l1_loss(outputs[:, 1], batch_y[:, 1]).item() * batch_X.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                val_mae = (val_mae_sp + val_mae_dp) / (2 * len(val_loader.dataset))
                val_losses.append(val_loss)
                val_mae_losses.append(val_mae)
                
                # 输出本轮结果（与蒸馏训练保持一致的格式）
                print(f"\nEpoch {epoch + 1} 摘要:")
                print(f"  Huber损失: {train_loss:.4f} (优化目标)")
                print(f"  训练MAE: {train_mae:.2f} mmHg (对比指标)")
                print(f"  验证MAE: {val_mae:.2f} mmHg")
                print(f"  收缩压MAE: {val_mae_sp/len(val_loader.dataset):.2f} mmHg | 舒张压MAE: {val_mae_dp/len(val_loader.dataset):.2f} mmHg")
                
                # 保存最佳模型
                if val_mae < best_mae:
                    best_mae = val_mae
                    with ema.average_parameters():
                        torch.save(student.state_dict(), student_path)
                    print("==> 保存EMA参数的最佳学生模型！")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"早停: {config.patience}个epoch没有改进")
                        break
                
                # 绘制训练曲线 - 与蒸馏训练保持一致的风格（固定文件名更新）
                try:
                    # MAE对比图（主要关注）
                    plt.figure(figsize=(10, 6))
                    epochs_range = range(1, len(train_mae_losses) + 1)
                    plt.plot(epochs_range, train_mae_losses, 'b-', label='Train MAE', linewidth=2, marker='o', markersize=2)
                    plt.plot(epochs_range, val_mae_losses, 'r-', label='Val MAE', linewidth=2, marker='s', markersize=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('MAE (mmHg)')
                    plt.title('Direct Student Training - MAE Comparison')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 添加最佳MAE标注
                    if len(val_mae_losses) > 0:
                        try:
                            best_epoch = int(np.argmin(val_mae_losses))
                            best_val_mae = float(val_mae_losses[best_epoch])
                            plt.text(0.02, 0.98, f'Best Val MAE: {best_val_mae:.2f} mmHg (Epoch {best_epoch+1})',
                                   transform=plt.gca().transAxes, fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                   verticalalignment='top')
                        except Exception as e:
                            print(f"标注添加失败: {e}")
                    
                    # 固定文件名，每次覆盖更新
                    mae_fig_name = f'models/direct_training_mae-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png'
                    plt.savefig(mae_fig_name, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Huber损失图（优化过程监控）
                    plt.figure(figsize=(10, 5))
                    plt.plot(epochs_range, train_losses, 'b-', label='Train Huber Loss', linewidth=2)
                    plt.plot(epochs_range, val_losses, 'r-', label='Val Huber Loss', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Huber Loss')
                    plt.title('Direct Student Training - Huber Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 固定文件名，每次覆盖更新
                    loss_fig_name = f'models/direct_training_loss-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png'
                    plt.savefig(loss_fig_name, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as plot_error:
                    print(f"绘图过程发生错误: {plot_error}")
                    print("跳过绘图，继续训练...")
                    plt.close('all')
        
        # 最终结果显示
        print(f"\n训练完成！最终结果:")
        print(f"  最佳验证MAE: {best_mae:.2f} mmHg")
        print(f"  总训练轮数: {len(train_losses)}")
        
        print("\n直接训练学生模型完成！")
        print(f"最佳模型保存在: {student_path}")
        print(f"训练曲线保存为:")
        print(f"  - MAE对比图: models/direct_training_mae-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png")
        print(f"  - Huber损失图: models/direct_training_loss-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 即使出错也关闭所有图表
        plt.close('all')


if __name__ == "__main__":
    train_student_direct() 