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
    def __init__(self, config):
        super().__init__()
        self.delta = config.huber_delta
        self.sp_weight = config.sp_weight

    def huber_loss(self, pred, target):
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear


    def forward(self, pred, target):

        if isinstance(pred, tuple):
            pred = pred[0]
            
        sp_loss = self.huber_loss(pred[:, 0], target[:, 0]).mean()
        dp_loss = self.huber_loss(pred[:, 1], target[:, 1]).mean()
        return self.sp_weight * sp_loss + (1 - self.sp_weight) * dp_loss

def train_teacher():
    config = Config()


    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")


    train_losses = []
    val_losses = []


    config.input_dim = X_train.shape[1]
    print("当前输入维度:", config.input_dim)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))  # 验证集

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, pin_memory=True)  # 验证集加载器


    model = BloodPressureTeacher(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.max_lr,
        total_steps=config.epochs * len(train_loader),
        pct_start=config.pct_start
    )


    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    criterion = WeightedHuberLoss(config)
    best_mae = float('inf')

    for epoch in range(config.epochs):

        model.train()
        train_loss = 0.0
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
            progress_bar.set_postfix({'loss': loss.item()})


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


        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = (val_mae_sp + val_mae_dp) / (2 * len(val_loader.dataset))


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} mmHg")


        if val_mae < best_mae:
            best_mae = val_mae
            with ema.average_parameters():
                torch.save(model.state_dict(), f"models/teacher_best-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.pth")
            print("==> New best model saved with EMA parameters!")
        

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('teacher training loss')
        plt.legend()
        plt.savefig(f'models/teacher_training_loss-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png')
        plt.close()

if __name__ == "__main__":
    train_teacher()
