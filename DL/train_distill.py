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
        self.feature_adapter = nn.Linear(1024, 256)

        nn.init.xavier_uniform_(self.feature_adapter.weight)
        nn.init.zeros_(self.feature_adapter.bias)
        

        self.mid_feature_adapter = nn.Linear(768, 256)
        nn.init.xavier_uniform_(self.mid_feature_adapter.weight)
        nn.init.zeros_(self.mid_feature_adapter.bias)
        

        self.sp_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

    def huber_loss(self, pred, target):
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        sp_loss = 0.5 * quadratic[:, 0] ** 2 + self.delta * (abs_error[:, 0] - quadratic[:, 0])
        dp_loss = 0.5 * quadratic[:, 1] ** 2 + self.delta * (abs_error[:, 1] - quadratic[:, 1])
        return self.sp_weight * sp_loss + self.dp_weight * dp_loss

    def forward(self, student_outputs, teacher_outputs, targets):
        try:
            student_pred, student_feats = student_outputs
            teacher_pred, teacher_feat = teacher_outputs

            last_student_feat = student_feats[-1]
            adapted_student_feat = self.feature_adapter(last_student_feat)

            attention_weights = self.sp_attention(adapted_student_feat)
            attended_feature = adapted_student_feat * attention_weights

            main_feature_loss = F.mse_loss(attended_feature, teacher_feat)

            mid_feature_loss = 0.0
            if len(student_feats) > 1:
                try:
                    mid_student_feat = student_feats[-2]
                    adapted_mid_feat = self.mid_feature_adapter(mid_student_feat)
                    mid_feature_loss = F.mse_loss(adapted_mid_feat, teacher_feat)
                except Exception as e:
                    mid_feature_loss = torch.tensor(0.0, device=student_pred.device)

            feature_loss = main_feature_loss + 0.5 * mid_feature_loss
            soft_loss = F.kl_div(
                F.log_softmax(student_pred / self.temp, dim=1),
                F.softmax(teacher_pred.detach() / self.temp, dim=1),
                reduction='batchmean'
            ) * (self.temp ** 2)

            target_loss = self.huber_loss(student_pred, targets).mean()

            total_loss = (
                self.alpha * target_loss + 
                self.soft_loss_weight * (1 - self.alpha) * soft_loss +
                self.feature_loss_weight * (1 - self.alpha) * feature_loss
            )
            
            return total_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)


def train_distill():
    config = Config()
    os.makedirs("models", exist_ok=True)

    train_losses = []
    val_losses = []

    X_train = np.load("data/processed/X_train.npy")
    selected_features = np.load("data/processed/selected_features.npy", allow_pickle=True)
    config.input_dim = X_train.shape[1]
    print(f"Number of input features: {config.input_dim}")

    teacher = BloodPressureTeacher(config).to(config.device)
    try:
        teacher.load_state_dict(torch.load("models/teacher_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", map_location=config.device))
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Teacher model not found - please train it first")
        return
    
    teacher.eval()
    with torch.no_grad():
        print("Now generating teacher model features...")
        X_tensor = torch.FloatTensor(X_train).to(config.device)

        batch_size = 1024
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        all_teacher_preds = []
        all_teacher_feats = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            batch_X = X_tensor[start_idx:end_idx]
            

            teacher.train()
            outputs = teacher(batch_X)
            teacher.eval()
            
            if isinstance(outputs, tuple):
                pred, layer_outputs = outputs
                feat = layer_outputs.get('layer3', None)
                if feat is None:
                    print("Warning: layer3 features not found; attempting to use other layers.")
                    for layer_name in ['layer2', 'layer1']:
                        feat = layer_outputs.get(layer_name, None)
                        if feat is not None:
                            print(f"Selected {layer_name} for features")
                            break
            else:
                pred = outputs
                feat = None
                
            all_teacher_preds.append(pred.cpu())
            if feat is not None:
                if len(feat.shape) == 4:
                    feat = torch.mean(feat, dim=[2, 3])
                all_teacher_feats.append(feat.cpu())
            
        teacher_pred = torch.cat(all_teacher_preds, dim=0)
        if len(all_teacher_feats) > 0:
            teacher_feat = torch.cat(all_teacher_feats, dim=0)
            teacher_outputs = (teacher_pred, teacher_feat)
            print(f"Finished generating teacher features (shape: {teacher_feat.shape})")
        else:
            teacher_feat = torch.zeros((len(teacher_pred), 512), device=teacher_pred.device)
            teacher_outputs = (teacher_pred, teacher_feat)


    student = MobileBPStudent(config).to(config.device)
    print(f"Finished generating teacher features (shape: {teacher_feat.shape})")


    try:
        X_warmup = torch.FloatTensor(X_train[:1000]).to(config.device)
        y_warmup = torch.FloatTensor(np.load("data/processed/y_train.npy")[:1000]).to(config.device)
        

        warmup_criterion = nn.MSELoss().to(config.device)
        warmup_optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        student.train()

        for i in range(3):
            warmup_optimizer.zero_grad()
            pred, _ = student(X_warmup)
            loss = warmup_criterion(pred, y_warmup)
            loss.backward()
            warmup_optimizer.step()
            print(f"[Warmup] Epoch {i + 1}/3 | Loss: {loss.item():.4f}")
            

    except Exception as e:
        print(f"[Warning] Warmup error: {e}, continuing training anyway")

    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(np.load("data/processed/y_train.npy")),
        teacher_outputs[0],
        teacher_outputs[1]
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )


    param_groups = [
        {'params': [], 'weight_decay': config.weight_decay * 1.5},
        {'params': [], 'weight_decay': config.weight_decay}
    ]
    

    for name, param in student.named_parameters():
        if 'classifier' in name or 'fc' in name or 'head' in name:
            param_groups[1]['params'].append(param)
        else:
            param_groups[0]['params'].append(param)
    

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.max_lr,
        eps=1e-6
    )


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr * 1.2,
        total_steps=config.epochs * len(loader),
        pct_start=0.25,
        div_factor=20,
        final_div_factor=20000,
        anneal_strategy='cos'
    )


    ema = ExponentialMovingAverage(student.parameters(), decay=0.996)


    criterion = EnhancedDistillLoss(config).to(config.device)

    best_mae = float('inf')
    patience_counter = 0

    print("[Verification] Testing student model forward propagation...")
    student.eval()
    with torch.no_grad():
        try:
            test_batch, _, test_teacher_pred, test_teacher_feat = next(iter(loader))
            test_batch = test_batch.to(config.device)
            student_outputs, student_features = student(test_batch)
            for i, feat in enumerate(student_features):
                print(f"{feat.shape}", end=", " if i < len(student_features) - 1 else "]\n")
                

            last_feat = student_features[-1].to(config.device)
            test_teacher_feat = test_teacher_feat.to(config.device)
            adapted_feat = criterion.feature_adapter(last_feat)
            print(f"[Shapes] Student (raw): {last_feat.shape} | Student (adapted): {adapted_feat.shape} | Teacher: {test_teacher_feat.shape}")
            print("[Success] Forward propagation validated!")
        except Exception as e:
            print(f"[Warning] Forward propagation test failed: {e}")
            print("[Recovery] Trying to proceed with training...")
    
    for epoch in range(config.epochs):
        student.train()
        total_loss = 0.0
        progress_bar = tqdm(loader, desc=f" Epoch {epoch + 1}/{config.epochs}")

        for batch_idx, (X, y_true, y_teacher_pred, y_teacher_feat) in enumerate(progress_bar):
            X = X.to(config.device)
            y_true = y_true.to(config.device)
            y_teacher = (y_teacher_pred.to(config.device), y_teacher_feat.to(config.device))
            

            if batch_idx == 0 and epoch == 0:
                print(f"\n批次数据形状: X={X.shape}, y_true={y_true.shape}")
                print(f"教师预测形状: {y_teacher_pred.shape}, 教师特征形状: {y_teacher_feat.shape}")

            optimizer.zero_grad()
            

            try:
                student_out = student(X)
                loss = criterion(student_out, y_teacher, y_true)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                ema.update()
                scheduler.step()
                
                total_loss += loss.item() * X.size(0)
                progress_bar.set_postfix({'loss': loss.item()})

            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                if batch_idx == 0:
                    print(f"Input X statistics: min={X.min().item()}, max={X.max().item()}, mean={X.mean().item()}")

                    try:
                        outputs, feats = student(X)
                        print(f"Student output shape: {outputs.shape}")
                        print(f"Student final feature shape: {feats[-1].shape}")
                        print(f"Teacher feature shape: {y_teacher[1].shape}")
                        adapted = criterion.feature_adapter(feats[-1])
                        print(f"Adapted feature shape: {adapted.shape}")
                    except Exception as inner_e:
                        print(f"Debug analysis failed: {inner_e}")
                if epoch == 0 and batch_idx == 0:
                    print("Error occurred in first batch, but attempting to continue training")

                continue


        student.eval()
        X_val = torch.FloatTensor(np.load("data/processed/X_val.npy")).to(config.device)
        y_val = torch.FloatTensor(np.load("data/processed/y_val.npy")).to(config.device)

        with torch.no_grad():
            try:
                preds, _ = student(X_val)
                

                val_mae = F.l1_loss(preds, y_val).item()
                

                sp_mae = F.l1_loss(preds[:, 0], y_val[:, 0]).item()
                dp_mae = F.l1_loss(preds[:, 1], y_val[:, 1]).item()
                
                avg_loss = total_loss / len(loader.dataset)
                val_loss = val_mae


                train_losses.append(avg_loss)
                val_losses.append(val_loss)

                print(f"\nEpoch {epoch + 1} summary:")
                print(f"  Training loss: {avg_loss:.4f}")
                print(f"  Validation MAE: {val_mae:.2f} mmHg")
                print(f"  Systolic MAE: {sp_mae:.2f} mmHg | Diastolic MAE: {dp_mae:.2f} mmHg")

                sp_dp_ratio = sp_mae / dp_mae if dp_mae > 0 else float('inf')
                print(f"  Systolic/Diastolic error ratio: {sp_dp_ratio:.2f}x")
                
                if val_mae < best_mae:
                    best_mae = val_mae
                    patience_counter = 0
                    with ema.average_parameters():

                        torch.save(student.state_dict(), f"models/student_best-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.pth")
                        

                        try:
                            dummy_input = torch.randn(1, config.input_dim, device=config.device)
                            torch.onnx.export(
                                student, 
                                dummy_input, 
                                f"models/student_mobile-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.onnx",
                                export_params=True,
                                opset_version=12,
                                input_names=['input'],
                                output_names=['output'],
                                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                            )
                        except Exception as e:
                            print(f"[Error] Failed to export model to ONNX: {e}")
                    print("==> Checkpointing student model with optimal EMA weights")
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f"[Early Stopping] No validation improvement for {config.patience} epochs")
                        break


                fig_name = f"models/distill_training_loss-distill_alpha={config.distill_alpha}-epochs={config.epochs}-batch_size={config.batch_size}-lr={config.max_lr}-distill_temp={config.distill_temp}-huber_delta={config.huber_delta}-feature_loss_weight={config.feature_loss_weight}-soft_loss_weight={config.soft_loss_weight}-sp_weight={config.sp_weight}-dp_weight={config.dp_weight}.png"
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='train loss')
                plt.plot(val_losses, label='val loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('distill training loss')
                plt.legend()
                plt.savefig(fig_name)
                plt.close()
                        
            except Exception as e:
                print(f"Validation phase error: {e}")
                continue


if __name__ == "__main__":
    train_distill()
