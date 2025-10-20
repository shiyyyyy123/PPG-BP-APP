"""
特征蒸馏与分布式回归蒸馏权重对比实验

本脚本用于比较不同feature_loss_weight和soft_loss_weight组合下的蒸馏模型性能。

注意：
- soft_loss_weight 现在指的是“分布式回归蒸馏”的权重，不是错误的softmax方法
- 分布式回归蒸馏：将血压预测值转换为高斯分布，然后计算KL散度
- 这比原始的“血压值softmax”方法更适合回归任务
"""

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
import time

# 组合参数——特征蒸馏 vs 分布式回归蒸馏的权重对比
feature_loss_weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]  # 特征蒸馏权重
soft_loss_weights =   [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # 分布式回归蒸馏权重


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
        start_time = time.time()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()
        inference_time = (time.time() - start_time) * 1000
        per_sample_time = inference_time / len(X)
    
    sp_mae = mean_absolute_error(y[:, 0], preds[:, 0])
    dp_mae = mean_absolute_error(y[:, 1], preds[:, 1])
    mae = mean_absolute_error(y, preds)
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

def plot_performance_comparison(results_dict, save_path):
    """绘制性能对比柱状图（严格分组，避免遮挡）"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['MAE', 'SP MAE', 'DP MAE', 'High SP MAE']
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)
    x = np.arange(n_metrics)
    group_width = 0.8  # 每组的总宽度
    width = group_width / n_models  # 每个柱子的宽度
    plt.figure(figsize=(max(12, 2*n_metrics), 2+1.5*n_models))
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4', '#FF7F00', '#000000', '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']
    for i, name in enumerate(model_names):
        values = [results_dict[name][m] for m in metrics]
        # 使每组的柱子严格在[组中心-group_width/2, 组中心+group_width/2]范围内
        offsets = x - group_width/2 + width/2 + i*width
        plt.bar(offsets, values, width, label=name, color=colors[i%len(colors)])
        for j, v in enumerate(values):
            plt.text(offsets[j], v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Error (mmHg)', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_comparison_table(results_dict):
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
    df.to_csv('results/feature_soft_comparison.csv', index=False, encoding='utf-8-sig')
    return df

def main():
    print("开始比较不同feature_loss_weight和soft_loss_weight下的蒸馏模型性能...")
    os.makedirs("results", exist_ok=True)
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]
    device = config.device
    print(f"使用设备: {device}")
    # 定义要比较的模型
    model_paths = {
        "Teacher": ("teacher_aug_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", BloodPressureTeacher),
        "Direct": ("student_direct_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", MobileBPStudent)
    }
    # 添加不同feature_loss_weight和soft_loss_weight组合的学生模型
    for f, s in zip(feature_loss_weights, soft_loss_weights):
        name = f"Distill(flw={f},slw={s})"
        path = f"student_best-distill_alpha=0.65-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight={f}-soft_loss_weight={s}-sp_weight=0.7-dp_weight=0.3.pth"
        model_paths[name] = (path, MobileBPStudent)
    # 加载和评估模型
    results_dict = {}
    for name, (path, model_class) in model_paths.items():
        full_path = os.path.join("models", path)
        model = load_model(model_class, full_path, config, device)
        if model:
            results = evaluate_model(model, X_test, y_test, device, name)
            results_dict[name] = results
    if len(results_dict) < len(model_paths):
        print("部分模型加载失败，请确保所有必要的模型文件存在")
        return
    # 每个模型单独误差分布图
    for name, results in results_dict.items():
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sp_errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(sp_errors, bins=30, alpha=0.7, color='#2C7BB6')
        plt.xlabel('SP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - SP MAE Distribution')
        plt.subplot(1, 2, 2)
        dp_errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(dp_errors, bins=30, alpha=0.7, color='#D7191C')
        plt.xlabel('DP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - DP MAE Distribution')
        plt.tight_layout()
        plt.savefig(f'results/feature_soft_error_distribution_{name}.png', dpi=300)
        plt.close()
    plot_performance_comparison(results_dict, 'results/feature_soft_performance.png')
    create_comparison_table(results_dict)
    print("\n比较完成！所有结果已保存至 results 目录")

if __name__ == "__main__":
    main() 