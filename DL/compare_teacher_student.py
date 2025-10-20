import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from models.teacher_model import BloodPressureTeacher
from models.student_model import MobileBPStudent
from configs import Config
import os
import pandas as pd
import seaborn as sns



def load_model(model_class, model_path, config, device):
    """加载模型"""
    model = model_class(config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败 {model_path}: {e}")
        return None

def evaluate_model(model, X, y, device, model_name="模型"):
    """评估模型性能"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        inputs = torch.FloatTensor(X).to(device)
        
        # 计时开始
        start_time = time.time()
        outputs = model(inputs)
        
        # 如果是元组(预测值, 特征)，提取预测值
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        per_sample_time = inference_time / len(X)
    
    # 计算评估指标
    sp_mae = mean_absolute_error(y[:, 0], preds[:, 0])
    dp_mae = mean_absolute_error(y[:, 1], preds[:, 1])
    mae = mean_absolute_error(y, preds)
    
    # 计算不同血压范围的误差
    high_sp_mask = y[:, 0] > 140
    normal_sp_mask = (y[:, 0] >= 120) & (y[:, 0] <= 140)
    low_sp_mask = y[:, 0] < 120
    
    high_sp_mae = mean_absolute_error(y[high_sp_mask, 0], preds[high_sp_mask, 0]) if np.sum(high_sp_mask) > 0 else np.nan
    normal_sp_mae = mean_absolute_error(y[normal_sp_mask, 0], preds[normal_sp_mask, 0]) if np.sum(normal_sp_mask) > 0 else np.nan
    low_sp_mae = mean_absolute_error(y[low_sp_mask, 0], preds[low_sp_mask, 0]) if np.sum(low_sp_mask) > 0 else np.nan
    
    results = {
        'model_name': model_name,
        'mae': mae,
        'sp_mae': sp_mae,
        'dp_mae': dp_mae,
        'high_sp_mae': high_sp_mae,
        'normal_sp_mae': normal_sp_mae,
        'low_sp_mae': low_sp_mae,
        'inference_time': per_sample_time,
        'predictions': preds,
        'true_values': y
    }
    
    return results

def plot_error_distribution(teacher_results, student_results, save_path):
    """绘制误差分布对比图"""
    plt.figure(figsize=(15, 6))
    
    # 收缩压误差分布
    plt.subplot(1, 2, 1)
    teacher_sp_errors = np.abs(teacher_results['true_values'][:, 0] - teacher_results['predictions'][:, 0])
    student_sp_errors = np.abs(student_results['true_values'][:, 0] - student_results['predictions'][:, 0])
    
    plt.hist(teacher_sp_errors, bins=30, alpha=0.5, label=f'Teacher Model (MAE={teacher_results["sp_mae"]:.2f})')
    plt.hist(student_sp_errors, bins=30, alpha=0.5, label=f'Student Model (MAE={student_results["sp_mae"]:.2f})')
    plt.xlabel('Systolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Systolic Pressure Error Distribution')
    plt.legend()
    
    # 舒张压误差分布
    plt.subplot(1, 2, 2)
    teacher_dp_errors = np.abs(teacher_results['true_values'][:, 1] - teacher_results['predictions'][:, 1])
    student_dp_errors = np.abs(student_results['true_values'][:, 1] - student_results['predictions'][:, 1])
    
    plt.hist(teacher_dp_errors, bins=30, alpha=0.5, label=f'Teacher Model (MAE={teacher_results["dp_mae"]:.2f})')
    plt.hist(student_dp_errors, bins=30, alpha=0.5, label=f'Student Model (MAE={student_results["dp_mae"]:.2f})')
    plt.xlabel('Diastolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Diastolic Pressure Error Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance_comparison(teacher_results, student_results, save_path):
    """绘制性能对比柱状图"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High BP MAE']
    
    teacher_values = [teacher_results[m] for m in metrics]
    student_values = [student_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, teacher_values, width, label='Teacher Model', color='#2C7BB6')
    plt.bar(x + width/2, student_values, width, label='Student Model', color='#D7191C')
    
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Error (mmHg)')
    plt.title('Teacher vs Student Model Performance')
    plt.xticks(x, metric_labels)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(teacher_values):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center')
    for i, v in enumerate(student_values):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_comparison_table(teacher_results, student_results):
    """创建性能对比表格"""
    data = {
        '模型': ['教师模型', '学生模型'],
        '总体MAE': [teacher_results['mae'], student_results['mae']],
        '收缩压MAE': [teacher_results['sp_mae'], student_results['sp_mae']],
        '舒张压MAE': [teacher_results['dp_mae'], student_results['dp_mae']],
        '高血压MAE': [teacher_results['high_sp_mae'], student_results['high_sp_mae']],
        '推理时间(ms/样本)': [teacher_results['inference_time'], student_results['inference_time']]
    }
    
    df = pd.DataFrame(data)
    print("\n模型性能对比表:")
    print(df.to_string(index=False))
    
    # 保存到CSV
    df.to_csv('results/teacher_student_comparison.csv', index=False, encoding='utf-8-sig')
    return df

def main():
    """主函数：比较教师模型和蒸馏学生模型的性能"""
    print("开始比较教师模型和蒸馏学生模型的性能...")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 加载配置和数据
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]
    
    device = config.device
    print(f"使用设备: {device}")
    
    # 加载模型
    teacher_model = load_model(BloodPressureTeacher, "models/teacher_best.pth", config, device)
    student_model = load_model(MobileBPStudent, "models/student_best.pth", config, device)
    
    if teacher_model is None or student_model is None:
        print("模型加载失败，请确保模型文件存在")
        return
    
    # 评估模型
    teacher_results = evaluate_model(teacher_model, X_test, y_test, device, "教师模型")
    student_results = evaluate_model(student_model, X_test, y_test, device, "学生模型")
    
    # 生成可视化比较
    plot_error_distribution(teacher_results, student_results, 'results/error_distribution_comparison.png')
    plot_performance_comparison(teacher_results, student_results, 'results/performance_comparison.png')
    
    # 创建对比表格
    comparison_df = create_comparison_table(teacher_results, student_results)
    
    # 计算模型大小
    teacher_size = os.path.getsize("models/teacher_best.pth") / (1024 * 1024)  # MB
    student_size = os.path.getsize("models/student_best.pth") / (1024 * 1024)  # MB
    
    # 输出总结报告
    print("\n性能对比总结:")
    print(f"模型大小比较:")
    print(f"- 教师模型: {teacher_size:.2f} MB")
    print(f"- 学生模型: {student_size:.2f} MB")
    print(f"- 压缩率: {teacher_size/student_size:.2f}x")
    
    print(f"\n预测性能比较:")
    print(f"- 总体MAE差异: {student_results['mae'] - teacher_results['mae']:.2f} mmHg")
    print(f"- 收缩压MAE差异: {student_results['sp_mae'] - teacher_results['sp_mae']:.2f} mmHg")
    print(f"- 舒张压MAE差异: {student_results['dp_mae'] - teacher_results['dp_mae']:.2f} mmHg")
    
    print(f"\n推理速度比较:")
    speedup = teacher_results['inference_time'] / student_results['inference_time']
    print(f"- 速度提升: {speedup:.2f}x")
    
    print("\n比较完成！所有结果已保存至 results 目录")

if __name__ == "__main__":
    import time
    main() 