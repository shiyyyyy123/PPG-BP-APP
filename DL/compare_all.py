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
    """Load model from checkpoint"""
    model = model_class(config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None


def evaluate_model(model, X, y, device, model_name="Model"):
    """Evaluate model performance metrics"""
    model.eval()
    results = {}

    with torch.no_grad():
        inputs = torch.FloatTensor(X).to(device)

        # Measure inference time
        start_time = time.time()
        outputs = model(inputs)

        # Handle both tuple and direct outputs
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()

        inference_time = (time.time() - start_time) * 1000  # milliseconds
        per_sample_time = inference_time / len(X)

    # Calculate MAE metrics
    sp_mae = mean_absolute_error(y[:, 0], preds[:, 0])  # Systolic
    dp_mae = mean_absolute_error(y[:, 1], preds[:, 1])  # Diastolic
    mae = mean_absolute_error(y, preds)  # Overall

    # Calculate BP range specific errors
    high_sp_mask = y[:, 0] > 140
    normal_sp_mask = (y[:, 0] >= 120) & (y[:, 0] <= 140)
    low_sp_mask = y[:, 0] < 120

    high_sp_mae = mean_absolute_error(y[high_sp_mask, 0], preds[high_sp_mask, 0]) if np.sum(
        high_sp_mask) > 0 else np.nan
    normal_sp_mae = mean_absolute_error(y[normal_sp_mask, 0], preds[normal_sp_mask, 0]) if np.sum(
        normal_sp_mask) > 0 else np.nan
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
    """Plot error distribution comparison"""
    plt.figure(figsize=(18, 8))  # Larger figure size
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4', '#FF7F00', '#000000']

    # Systolic BP error distribution
    plt.subplot(1, 2, 1)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                 label=f'{name} (MAE={results["sp_mae"]:.2f})')

    plt.xlabel('SP MAE (mmHg)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Systolic Blood Pressure Error Distribution', fontsize=14)
    plt.legend(fontsize=10)

    # Diastolic BP error distribution
    plt.subplot(1, 2, 2)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                 label=f'{name} (MAE={results["dp_mae"]:.2f})')

    plt.xlabel('DP MAE (mmHg)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Diastolic Blood Pressure Error Distribution', fontsize=14)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_comparison(results_dict, save_path):
    """Create performance comparison bar chart"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['MAE', 'SP MAE', 'DP MAE', 'High SP MAE']

    n_models = len(results_dict)
    x = np.arange(len(metrics))
    width = 0.12  # Narrower bars for more models

    plt.figure(figsize=(18, 8))  # Larger figure size
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4', '#FF7F00', '#000000']

    for i, (name, results) in enumerate(results_dict.items()):
        values = [results[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = plt.bar(x + offset, values, width, label=name, color=colors[i])

        # Add value labels with rotation
        for j, v in enumerate(values):
            plt.text(j + offset, v + 0.05, f'{v:.2f}',
                     ha='center', va='bottom',
                     fontsize=7, rotation=45)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Error (mmHg)', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    # Adjust layout for better visibility
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_inference_time_comparison(results_dict, save_path):
    """Plot inference time comparison"""
    names = list(results_dict.keys())
    times = [results['inference_time'] for results in results_dict.values()]

    plt.figure(figsize=(12, 6))  # Wider figure
    bars = plt.bar(names, times)

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Inference Time (ms/sample)', fontsize=12)
    plt.title('Model Inference Time Comparison', fontsize=14)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}ms', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')  # Rotated labels for better readability
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_comparison_table(results_dict):
    """Generate performance comparison table"""
    data = {
        'Model': [],
        'Overall MAE': [],
        'Systolic MAE': [],
        'Diastolic MAE': [],
        'High SP MAE': [],
        'Inference Time (ms/sample)': []
    }

    for name, results in results_dict.items():
        data['Model'].append(name)
        data['Overall MAE'].append(results['mae'])
        data['Systolic MAE'].append(results['sp_mae'])
        data['Diastolic MAE'].append(results['dp_mae'])
        data['High SP MAE'].append(results['high_sp_mae'])
        data['Inference Time (ms/sample)'].append(results['inference_time'])

    df = pd.DataFrame(data)
    print("\nModel Performance Comparison:")
    print(df.to_string(index=False))

    df.to_csv('results/distill_variants_comparison.csv', index=False, encoding='utf-8-sig')
    return df


def main():
    """Main function: Compare distillation models with different alpha values"""
    print("Comparing distillation models with different alpha values...")

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load configuration and data
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]

    device = config.device
    print(f"Using device: {device}")

    # Define models to compare
    model_paths = {
        "Teacher": (
        "teacher_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", BloodPressureTeacher),
        "Direct": (
        "student_direct_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", MobileBPStudent),
        "α=0.4": (
        "student_best-distill_alpha=0.4-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        MobileBPStudent),
        "α=0.5": (
        "student_best-distill_alpha=0.5-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        MobileBPStudent),
        "α=0.6": (
        "student_best-distill_alpha=0.6-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        MobileBPStudent),
        "α=0.65": (
        "student_best-distill_alpha=0.65-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        MobileBPStudent),
        "α=0.7": (
        "student_best-distill_alpha=0.7-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        MobileBPStudent)
    }

    # Load and evaluate models
    results_dict = {}
    for name, (path, model_class) in model_paths.items():
        full_path = os.path.join("models", path)
        model = load_model(model_class, full_path, config, device)
        if model:
            results = evaluate_model(model, X_test, y_test, device, name)
            results_dict[name] = results

    if len(results_dict) < len(model_paths):
        print("Some models failed to load - please ensure all required model files exist")
        return

    # Generate individual error distribution plots
    for name, results in results_dict.items():
        plt.figure(figsize=(12, 5))
        # Systolic BP error distribution
        plt.subplot(1, 2, 1)
        sp_errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(sp_errors, bins=30, alpha=0.7, color='#2C7BB6')
        plt.xlabel('SP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - Systolic BP Error Distribution')
        # Diastolic BP error distribution
        plt.subplot(1, 2, 2)
        dp_errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(dp_errors, bins=30, alpha=0.7, color='#D7191C')
        plt.xlabel('DP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - Diastolic BP Error Distribution')
        plt.tight_layout()
        plt.savefig(f'results/distill_variants_error_distribution_{name}.png', dpi=300)
        plt.close()

    # Generate comparison plots
    plot_performance_comparison(results_dict, 'results/distill_variants_performance.png')
    plot_inference_time_comparison(results_dict, 'results/distill_variants_inference_time.png')

    # Create comparison table
    comparison_df = create_comparison_table(results_dict)

    # Calculate and print improvement analysis
    print("\nKnowledge Distillation Improvement Analysis:")
    base_model = results_dict["Direct"]
    teacher_model = results_dict["Teacher"]

    alphas = ['0.4', '0.5', '0.6', '0.65', '0.7']
    for alpha in alphas:
        distill_model = results_dict[f"α={alpha}"]
        print(f"\n=== Analysis for α={alpha} distilled model ===")

        # Comparison with direct-trained model
        print("Improvement over direct-trained model:")
        for metric, name in zip(['mae', 'sp_mae', 'dp_mae', 'high_sp_mae'],
                                ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High SP MAE']):
            improvement = (base_model[metric] - distill_model[metric]) / base_model[metric] * 100
            print(f"{name}: {improvement:.2f}% improvement")

        # Comparison with teacher model
        print("\nPerformance relative to teacher model:")
        for metric, name in zip(['mae', 'sp_mae', 'dp_mae', 'high_sp_mae'],
                                ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High SP MAE']):
            ratio = distill_model[metric] / teacher_model[metric] * 100
            print(f"{name}: {ratio:.2f}% of teacher's error")

    print("\nComparison completed! All results saved to 'results' directory")


if __name__ == "__main__":
    main()