import torch
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.teacher_model import BloodPressureTeacher
from models.student_model import MobileBPStudent
from configs import Config
import time
import matplotlib.pyplot as plt

def load_and_evaluate_model(model_path, model_class, X_test, y_test, config, model_name="Model"):
    """加载模型并在测试集上评估性能"""
    model = model_class(config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    
    # 预热GPU（如果使用GPU）
    if config.device != 'cpu':
        dummy_input = torch.FloatTensor(X_test[:1]).to(config.device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
    
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(config.device)
        
        # 多次推理取平均时间，提高准确性
        num_runs = 50
        total_time = 0
        
        for _ in range(num_runs):
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # 毫秒
        
        # 计算平均推理时间（不包含数据转换）
        inference_time = total_time / num_runs
        
        # 获取预测结果（单独计算，不计入推理时间）
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()
    sp_mae = mean_absolute_error(y_test[:, 0], preds[:, 0])
    dp_mae = mean_absolute_error(y_test[:, 1], preds[:, 1])
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'sp_mae': sp_mae,
        'dp_mae': dp_mae,
        'r2': r2,
        'inference_time': inference_time,
        'model_path': model_path,
        'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'predictions': preds,
        'true_values': y_test
    }

def plot_predictions(results, plot_title, save_path=None):
    preds = results['predictions']
    true = results['true_values']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(true[:, 0], preds[:, 0], alpha=0.5)
    ax1.plot([true[:, 0].min(), true[:, 0].max()], [true[:, 0].min(), true[:, 0].max()], 'r--')
    ax1.set_xlabel('Actual Systolic Pressure (mmHg)')
    ax1.set_ylabel('Predicted Systolic Pressure (mmHg)')
    ax1.set_title(f'Systolic Pressure Prediction (MAE: {results["sp_mae"]:.2f} mmHg)')
    ax2.scatter(true[:, 1], preds[:, 1], alpha=0.5)
    ax2.plot([true[:, 1].min(), true[:, 1].max()], [true[:, 1].min(), true[:, 1].max()], 'r--')
    ax2.set_xlabel('Actual Diastolic Pressure (mmHg)')
    ax2.set_ylabel('Predicted Diastolic Pressure (mmHg)')
    ax2.set_title(f'Diastolic Pressure Prediction (MAE: {results["dp_mae"]:.2f} mmHg)')
    fig.suptitle(plot_title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"预测散点图已保存到: {save_path}")
    plt.close()

def main():
    print("=" * 50)
    print("血压预测模型最终测试脚本")
    print("=" * 50)
    os.makedirs("results", exist_ok=True)
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    print(f"测试集大小: {X_test.shape}")
    config.input_dim = X_test.shape[1]
    device = config.device
    print(f"使用设备: {device}")
    # 模型路径
    no_aug_teacher_path = "models/teacher_no_aug_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    aug_teacher_path = "models/teacher_aug_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    student_direct_path = "models/student_direct_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    student_distill_path = "models/student_best-layers=L1_L2_L3_Final-distill_alpha=0.4-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth"
    # 评估教师模型
    no_aug_teacher_results = None
    if os.path.exists(no_aug_teacher_path):
        no_aug_teacher_results = load_and_evaluate_model(
            no_aug_teacher_path, BloodPressureTeacher, X_test, y_test, config, "Teacher Model (ResNet) (No Augmentation)")
        plot_predictions(no_aug_teacher_results, "Teacher Model (ResNet) (No Augmentation) Prediction Results", "results/no_aug_teacher_predictions.png")
    else:
        print("未找到教师模型(no_aug)权重，请先训练教师模型！")
    aug_teacher_results = None
    if os.path.exists(aug_teacher_path):
        aug_teacher_results = load_and_evaluate_model(
            aug_teacher_path, BloodPressureTeacher, X_test, y_test, config, "Teacher Model (ResNet) (Augmentation)")
        plot_predictions(aug_teacher_results, "Teacher Model (ResNet) (Augmentation) Prediction Results", "results/aug_teacher_predictions.png")
    else:
        print("未找到教师模型(aug)权重，请先训练教师模型！")
    # 评估直接学生模型
    student_direct_results = None
    if os.path.exists(student_direct_path):
        student_direct_results = load_and_evaluate_model(
            student_direct_path, MobileBPStudent, X_test, y_test, config, "Student Model (Direct)")
        plot_predictions(student_direct_results, "Student Model (Direct) Prediction Results", "results/student_direct_predictions.png")
    else:
        print("未找到直接训练学生模型权重，请先训练学生模型！")
    # 评估蒸馏学生模型
    student_distill_results = None
    if os.path.exists(student_distill_path):
        student_distill_results = load_and_evaluate_model(
            student_distill_path, MobileBPStudent, X_test, y_test, config, "Student Model (Distill)")
        plot_predictions(student_distill_results, "Student Model (Distill) Prediction Results", "results/student_distill_predictions.png")
    else:
        print("未找到蒸馏学生模型权重，请先训练蒸馏学生模型！")
    # 汇总对比
    print("\n模型性能对比:")
    results_list = [r for r in [no_aug_teacher_results, aug_teacher_results, student_direct_results, student_distill_results] if r is not None]
    print(results_list)
    if results_list:
        import pandas as pd
        df = pd.DataFrame([
            {
                '模型': r['model_name'],
                'MAE': r['mae'],
                'SP_MAE': r['sp_mae'],
                'DP_MAE': r['dp_mae'],
                'RMSE': r['rmse'],
                'R2': r['r2'],
                '推理时间(ms)': r['inference_time'],
                '模型大小(MB)': r['model_size_mb']
            }
            for r in results_list
        ])
        print(df.to_string(index=False))
        df.to_csv('results/final_model_comparison.csv', index=False, encoding='utf-8-sig')
    print("\n测试完成！所有结果已保存至 results 目录")

if __name__ == "__main__":
    main() 