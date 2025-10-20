import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from models.student_model import MobileBPStudent
from configs import Config
import os
import pandas as pd
import seaborn as sns
import time

def load_model(model_path, config, device):
    """加载模型"""
    model = MobileBPStudent(config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载失败: {e}")
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

def plot_error_distribution(results_dict, save_path):
    """绘制误差分布对比图"""
    plt.figure(figsize=(15, 6))
    colors = ['#2C7BB6', '#D7191C']  # 蓝、红
    
    # 收缩压误差分布
    plt.subplot(1, 2, 1)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                label=f'{name} (MAE={results["sp_mae"]:.2f})')
    
    plt.xlabel('Systolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Systolic Pressure Error Distribution')
    plt.legend()
    
    # 舒张压误差分布
    plt.subplot(1, 2, 2)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                label=f'{name} (MAE={results["dp_mae"]:.2f})')
    
    plt.xlabel('Diastolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Diastolic Pressure Error Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance_comparison(results_dict, save_path):
    """绘制性能对比柱状图"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High BP MAE']
    
    n_models = len(results_dict)
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    colors = ['#2C7BB6', '#D7191C']  # 蓝、红
    
    for i, (name, results) in enumerate(results_dict.items()):
        values = [results[m] for m in metrics]
        offset = (i - (n_models-1)/2) * width
        bars = plt.bar(x + offset, values, width, label=name, color=colors[i])
        
        # 添加数值标签
        for j, v in enumerate(values):
            plt.text(j + offset, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Error (mmHg)')
    plt.title('Student Model Variants Performance Comparison')
    plt.xticks(x, metric_labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_comparison_table(results_dict):
    """创建性能对比表格"""
    data = {
        '模型': [],
        '总体MAE': [],
        '收缩压MAE': [],
        '舒张压MAE': [],
        '高血压MAE': [],
        '推理时间(ms/样本)': []
    }
    
    for name, results in results_dict.items():
        data['模型'].append(name)
        data['总体MAE'].append(results['mae'])
        data['收缩压MAE'].append(results['sp_mae'])
        data['舒张压MAE'].append(results['dp_mae'])
        data['高血压MAE'].append(results['high_sp_mae'])
        data['推理时间(ms/样本)'].append(results['inference_time'])
    
    df = pd.DataFrame(data)
    print("\n模型性能对比表:")
    print(df.to_string(index=False))
    
    # 保存到CSV
    df.to_csv('results/student_variants_comparison.csv', index=False, encoding='utf-8-sig')
    return df

def main():
    """主函数：比较不同学生模型变体的性能"""
    print("开始比较不同学生模型变体的性能...")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 加载配置和数据
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]
    
    device = config.device
    print(f"使用设备: {device}")
    
    # 模型路径
    model_paths = {
        "Distilled Student": "models/student_best-distill_alpha=0.65-epochs=300-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        "Direct Trained Student": "models/student_direct_best-epochs=300-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    }
    
    # 加载和评估模型
    results_dict = {}
    for name, model_path in model_paths.items():
        model = load_model(model_path, config, device)
        if model:
            results = evaluate_model(model, X_test, y_test, device, name)
            results_dict[name] = results
    
    if len(results_dict) < 2:
        print("模型加载失败，请确保所有必要的模型文件存在")
        return
    
    # 生成可视化比较
    plot_error_distribution(results_dict, 'results/student_variants_error_distribution.png')
    plot_performance_comparison(results_dict, 'results/student_variants_performance.png')
    
    # 创建对比表格
    comparison_df = create_comparison_table(results_dict)
    
    # 计算和输出改进百分比
    base_model = results_dict["Direct Trained Student"]
    distill_model = results_dict["Distilled Student"]
    
    print("\n知识蒸馏改进分析:")
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_names = ['总体MAE', '收缩压MAE', '舒张压MAE', '高血压MAE']
    
    for metric, name in zip(metrics, metric_names):
        improvement = (base_model[metric] - distill_model[metric]) / base_model[metric] * 100
        print(f"{name}改进: {improvement:.2f}%")
    
    print("\n比较完成！所有结果已保存至 results 目录")

if __name__ == "__main__":
    main() 