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

def plot_error_distribution(results_dict, save_path):
    """绘制误差分布对比图"""
    plt.figure(figsize=(18, 8))  # 增加图表大小
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4', '#FF7F00', '#000000']
    
    # 收缩压误差分布
    plt.subplot(1, 2, 1)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                label=f'{name} (MAE={results["sp_mae"]:.2f})')
    
    plt.xlabel('SP MAE (mmHg)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('SP MAE Distribution', fontsize=14)
    plt.legend(fontsize=10)
    
    # 舒张压误差分布
    plt.subplot(1, 2, 2)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                label=f'{name} (MAE={results["dp_mae"]:.2f})')
    
    plt.xlabel('DP MAE (mmHg)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('DP MAE Distribution', fontsize=14)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_performance_comparison(results_dict, save_path):
    """绘制性能对比柱状图"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['MAE', 'SP MAE', 'DP MAE', 'High SP MAE']
    
    n_models = len(results_dict)
    x = np.arange(len(metrics))
    width = 0.12  # 减小柱状图宽度以适应更多模型
    
    plt.figure(figsize=(18, 8))  # 增加图表大小
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4', '#FF7F00', '#000000']
    
    for i, (name, results) in enumerate(results_dict.items()):
        values = [results[m] for m in metrics]
        offset = (i - (n_models-1)/2) * width
        bars = plt.bar(x + offset, values, width, label=name, color=colors[i])
        
        # 添加数值标签，调整字体大小和位置
        for j, v in enumerate(values):
            plt.text(j + offset, v + 0.05, f'{v:.2f}', 
                    ha='center', va='bottom', 
                    fontsize=7, rotation=45)
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Error (mmHg)', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # 调整布局以确保所有元素可见
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_inference_time_comparison(results_dict, save_path):
    """绘制推理时间对比图"""
    names = list(results_dict.keys())
    times = [results['inference_time'] for results in results_dict.values()]
    
    plt.figure(figsize=(12, 6))  # 增加图表宽度
    bars = plt.bar(names, times)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Inference Time (ms/sample)', fontsize=12)
    plt.title('Model Inference Time Comparison', fontsize=14)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')  # 调整标签角度和对齐方式
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
    
    df.to_csv('results/distill_variants_comparison.csv', index=False, encoding='utf-8-sig')
    return df

def main():
    """主函数：比较不同alpha值的蒸馏模型性能"""
    print("开始比较不同alpha值的蒸馏模型性能...")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 加载配置和数据
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]
    
    device = config.device
    print(f"使用设备: {device}")
    
    # 定义要比较的模型
    model_paths = {
        "Teacher": ("teacher_aug_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", BloodPressureTeacher),
        "Direct": ("student_direct_best-epochs=100-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", MobileBPStudent),
        "α=0.3": ("student_best-layers=L1_L2_L3_Final-distill_alpha=0.3-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth", MobileBPStudent),
        "α=0.4": ("student_best-layers=L1_L2_L3_Final-distill_alpha=0.4-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth", MobileBPStudent),
        "α=0.5": ("student_best-layers=L1_L2_L3_Final-distill_alpha=0.5-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth", MobileBPStudent),
        "α=0.6": ("student_best-layers=L1_L2_L3_Final-distill_alpha=0.6-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth", MobileBPStudent),
        "α=0.7": ("student_best-layers=L1_L2_L3_Final-distill_alpha=0.7-epochs=100-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=1.0-soft_loss_weight=0.0-sp_weight=0.7-dp_weight=0.3.pth", MobileBPStudent)
    }
    
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
    
    # 生成可视化比较
    for name, results in results_dict.items():
        plt.figure(figsize=(12, 5))
        # 收缩压误差分布
        plt.subplot(1, 2, 1)
        sp_errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(sp_errors, bins=30, alpha=0.7, color='#2C7BB6')
        plt.xlabel('SP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - SP MAE Distribution')
        # 舒张压误差分布
        plt.subplot(1, 2, 2)
        dp_errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(dp_errors, bins=30, alpha=0.7, color='#D7191C')
        plt.xlabel('DP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - DP MAE Distribution')
        plt.tight_layout()
        plt.savefig(f'results/distill_variants_error_distribution_{name}.png', dpi=300)
        plt.close()
    plot_performance_comparison(results_dict, 'results/distill_variants_performance.png')
    plot_inference_time_comparison(results_dict, 'results/distill_variants_inference_time.png')
    
    # 创建对比表格
    comparison_df = create_comparison_table(results_dict)
    
    # 计算和输出改进分析
    print("\n知识蒸馏改进分析:")
    base_model = results_dict["Direct"]
    teacher_model = results_dict["Teacher"]
    
    alphas = ['0.3', '0.4', '0.5', '0.6', '0.7']
    for alpha in alphas:
        distill_model = results_dict[f"α={alpha}"]
        print(f"\n=== α={alpha} 的蒸馏模型分析 ===")
        
        # 与直接训练模型比较
        print("相对于直接训练模型的改进:")
        for metric, name in zip(['mae', 'sp_mae', 'dp_mae', 'high_sp_mae'],
                              ['总体MAE', '收缩压MAE', '舒张压MAE', '高血压MAE']):
            improvement = (base_model[metric] - distill_model[metric]) / base_model[metric] * 100
            print(f"{name}: {improvement:.2f}%")
        
        # 与教师模型比较
        print("\n相对于教师模型的性能比:")
        for metric, name in zip(['mae', 'sp_mae', 'dp_mae', 'high_sp_mae'],
                              ['总体MAE', '收缩压MAE', '舒张压MAE', '高血压MAE']):
            ratio = distill_model[metric] / teacher_model[metric] * 100
            print(f"{name}: {ratio:.2f}%")
    
    print("\n比较完成！所有结果已保存至 results 目录")

if __name__ == "__main__":
    main() 