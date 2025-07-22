
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


class DirectLoss(nn.Module):
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
        sp_loss = self.huber_loss(pred[:, 0], target[:, 0]).mean()
        dp_loss = self.huber_loss(pred[:, 1], target[:, 1]).mean()
        return self.sp_weight * sp_loss + (1 - self.sp_weight) * dp_loss


def train_student_direct():
    config = Config()

    os.makedirs("models", exist_ok=True)
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    selected_features = np.load("data/processed/selected_features.npy", allow_pickle=True)
    config.input_dim = X_train.shape[1]
    student = MobileBPStudent(config).to(config.device)

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

    criterion = DirectLoss(config)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.max_lr,
        total_steps=config.epochs * len(train_loader),
        pct_start=config.pct_start
    )

    ema = ExponentialMovingAverage(student.parameters(), decay=0.999)

    best_mae = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    student_path = f"models/student_direct_best-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.pth"
    
    print(f"Starting training, maximum epochs: {config.epochs}")
    try:
        for epoch in range(config.epochs):
            student.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(config.device), batch_y.to(config.device)
                optimizer.zero_grad()
                outputs, _ = student(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                ema.update()
                train_loss += loss.item() * batch_X.size(0)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            

            student.eval()
            val_loss = 0.0
            val_mae_sp = 0.0
            val_mae_dp = 0.0
            
            with torch.no_grad(), ema.average_parameters():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.device), batch_y.to(config.device)
                    outputs, _ = student(batch_X)
                    
                    val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
                    val_mae_sp += F.l1_loss(outputs[:, 0], batch_y[:, 0]).item() * batch_X.size(0)
                    val_mae_dp += F.l1_loss(outputs[:, 1], batch_y[:, 1]).item() * batch_X.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                val_losses.append(val_loss)
                val_mae = (val_mae_sp + val_mae_dp) / (2 * len(val_loader.dataset))
                

                print(f"\nEpoch {epoch + 1}:")
                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  Val MAE: {val_mae:.2f} mmHg")
                

                if val_mae < best_mae:
                    best_mae = val_mae
                    with ema.average_parameters():
                        torch.save(student.state_dict(), student_path)
                    print("==> New best model saved with EMA parameters!")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"Early stop: {config.patience} epochs no improvement")
                        break
        

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('direct training loss')
        plt.legend()
        plt.savefig(f'models/direct_training_loss-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-huber_delta={config.huber_delta}-sp_weight={config.sp_weight}.png')
        plt.close()

        print("\nStandalone student model training complete!")
        print(f"Best model saved at: {student_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    train_student_direct() 