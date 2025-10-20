# train_teacher.py (兼容旧版PyTorch)
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.teacher_model import BloodPressureTeacher
from configs import Config
from tqdm import tqdm
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
import matplotlib.pyplot as plt

class WeightedHuberLoss(nn.Module):
    #加权Huber损失
    def __init__(self, config):
        super().__init__()
        self.delta = config.huber_delta
        self.sp_weight = config.sp_weight

    def huber_loss(self, pred, target):
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear

    #def huber_loss(self, pred, target):
    #    abs_error = torch.abs(pred - target)
    #    quadratic = torch.where(abs_error < self.delta, 0.5 * abs_error ** 2, 0.0)
    #    linear = torch.where(abs_error >= self.delta, self.delta * (abs_error - 0.5 * self.delta), 0.0)
    #    return quadratic + linear

    def forward(self, pred, target):

        if isinstance(pred, tuple):
            pred = pred[0]
            
        sp_loss = self.huber_loss(pred[:, 0], target[:, 0]).mean()
        dp_loss = self.huber_loss(pred[:, 1], target[:, 1]).mean()
        return self.sp_weight * sp_loss + (1 - self.sp_weight) * dp_loss

def train_teacher():
    config = Config()

    # 加载预处理数据 - 使用未增强的训练集
    X_train = np.load("data/processed/X_train_no_aug.npy")  # 改为未增强数据
    y_train = np.load("data/processed/y_train_no_aug.npy")  # 改为未增强数据
    X_val = np.load("data/processed/X_val.npy")  # 使用验证集
    y_val = np.load("data/processed/y_val.npy")  # 使用验证集

    # 记录训练和验证损失
    train_losses = []
    val_losses = []

    # 配置输入维度
    config.input_dim = X_train.shape[1]
    print("当前输入维度:", config.input_dim)
    print("使用未增强训练集进行教师模型训练（用于消融实验）")
    print(f"训练集大小: {X_train.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))  # 验证集

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, pin_memory=True)  # 验证集加载器

    # 初始化模型和优化器
    model = BloodPressureTeacher(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.max_lr,
        total_steps=config.epochs * len(train_loader),
        pct_start=config.pct_start
    )

    # 指数移动平均 (EMA)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    criterion = WeightedHuberLoss(config)
    best_mae = float('inf')

    # 记录MAE指标（用于绘图）
    train_mae_list = []
    val_mae_list = []

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae_sp = 0.0
        train_mae_dp = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for X, y in progress_bar:
            X, y = X.to(config.device), y.to(config.device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            scheduler.step()

            train_loss += loss.item() * X.size(0)
            
            # 计算训练MAE
            pred = outputs[0] if isinstance(outputs, tuple) else outputs
            train_mae_sp += F.l1_loss(pred[:, 0], y[:, 0]).item() * X.size(0)
            train_mae_dp += F.l1_loss(pred[:, 1], y[:, 1]).item() * X.size(0)
            
            progress_bar.set_postfix({'loss': loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae_sp = 0.0
        val_mae_dp = 0.0

        with torch.no_grad(), ema.average_parameters():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                outputs = model(X)
                
                val_loss += criterion(outputs, y).item() * X.size(0)

                pred = outputs[0] if isinstance(outputs, tuple) else outputs
                val_mae_sp += F.l1_loss(pred[:, 0], y[:, 0]).item() * X.size(0)
                val_mae_dp += F.l1_loss(pred[:, 1], y[:, 1]).item() * X.size(0)

        # 计算指标
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_mae = (train_mae_sp + train_mae_dp) / (2 * len(train_loader.dataset))
        val_mae = (val_mae_sp + val_mae_dp) / (2 * len(val_loader.dataset))
        val_mae_sp_avg = val_mae_sp / len(val_loader.dataset)
        val_mae_dp_avg = val_mae_dp / len(val_loader.dataset)

        # 记录损失和MAE
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train MAE: {train_mae:.2f} mmHg | Val MAE: {val_mae:.2f} mmHg")
        print(f"  收缩压MAE: {val_mae_sp_avg:.2f} mmHg | 舒张压MAE: {val_mae_dp_avg:.2f} mmHg")

        # 保存最佳模型 (包含EMA参数) - 更新文件名以区分消融实验
        if val_mae < best_mae:
            best_mae = val_mae
            with ema.average_parameters():
                torch.save(model.state_dict(), f"models/teacher_no_aug_best-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.pth")
            print("==> New best model saved with EMA parameters! (No Augmentation)")
        
        # 绘制训练曲线 - 参考蒸馏代码风格
        try:
            # MAE对比图（主图）
            plt.figure(figsize=(10, 6))
            epochs_range = range(1, len(train_mae_list) + 1)
            plt.plot(epochs_range, train_mae_list, 'b-', label='Train MAE', linewidth=2, marker='o', markersize=2)
            plt.plot(epochs_range, val_mae_list, 'r-', label='Val MAE', linewidth=2, marker='s', markersize=2)
            plt.xlabel('Epoch')
            plt.ylabel('MAE (mmHg)')
            plt.title('Teacher Model Training - MAE Comparison (No Augmentation)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加最佳MAE标注
            if len(val_mae_list) > 0:
                try:
                    best_epoch = int(np.argmin(val_mae_list))
                    best_val_mae = float(val_mae_list[best_epoch])
                    
                    plt.text(0.02, 0.98, f'Best Val MAE: {best_val_mae:.2f} mmHg (Epoch {best_epoch+1})',
                           transform=plt.gca().transAxes, fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                           verticalalignment='top')
                except Exception as e:
                    print(f"标注添加失败: {e}")
            
            plt.savefig(f'models/teacher_no_aug_training_mae-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 损失曲线图（辅助图）
            plt.figure(figsize=(10, 5))
            plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Teacher Model Training - Loss Curve (No Augmentation)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'models/teacher_no_aug_training_loss-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as plot_error:
            print(f"绘图过程发生错误: {plot_error}")
            print("跳过绘图，继续训练...")
            plt.close('all')

if __name__ == "__main__":
    train_teacher()
