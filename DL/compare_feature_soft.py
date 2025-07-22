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

# Parameter combinations to compare
feature_loss_weights = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
soft_loss_weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


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
        start_time = time.time()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        per_sample_time = inference_time / len(X)

    # Calculate evaluation metrics
    sp_mae = mean_absolute_error(y[:, 0], preds[:, 0])  # Systolic
    dp_mae = mean_absolute_error(y[:, 1], preds[:, 1])  # Diastolic
    mae = mean_absolute_error(y, preds)  # Overall

    # Calculate errors for different BP ranges
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


def plot_performance_comparison(results_dict, save_path):
    """Create grouped bar chart for performance comparison"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['MAE', 'SP MAE', 'DP MAE', 'High SP MAE']
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)

    # Plot configuration
    x = np.arange(n_metrics)
    group_width = 0.8  # Total width per group
    width = group_width / n_models  # Individual bar width

    plt.figure(figsize=(max(12, 2 * n_metrics), 2 + 1.5 * n_models))
    colors = ['#2C7BB6', '#D7191C', '#92C5DE', '#F4A582', '#4575B4',
              '#FF7F00', '#000000', '#66C2A5', '#FC8D62', '#8DA0CB',
              '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3']

    for i, name in enumerate(model_names):
        values = [results_dict[name][m] for m in metrics]
        # Position bars within group boundaries
        offsets = x - group_width / 2 + width / 2 + i * width
        plt.bar(offsets, values, width, label=name, color=colors[i % len(colors)])

        # Add value labels
        for j, v in enumerate(values):
            plt.text(offsets[j], v + 0.05, f'{v:.2f}',
                     ha='center', va='bottom',
                     fontsize=7, rotation=45)

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Error (mmHg)', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, metric_labels, fontsize=10)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_comparison_table(results_dict):
    """Generate performance comparison table"""
    data = {
        'Model': [],
        'Overall MAE': [],
        'Systolic MAE': [],
        'Diastolic MAE': [],
        'High BP MAE': [],
        'Inference Time (ms/sample)': []
    }

    for name, results in results_dict.items():
        data['Model'].append(name)
        data['Overall MAE'].append(results['mae'])
        data['Systolic MAE'].append(results['sp_mae'])
        data['Diastolic MAE'].append(results['dp_mae'])
        data['High BP MAE'].append(results['high_sp_mae'])
        data['Inference Time (ms/sample)'].append(results['inference_time'])

    df = pd.DataFrame(data)
    print("\nModel Performance Comparison:")
    print(df.to_string(index=False))
    df.to_csv('results/feature_soft_comparison.csv', index=False, encoding='utf-8-sig')
    return df


def main():
    """Main function: Compare distillation with different parameter combinations"""
    print("Comparing distilled models with different feature/soft loss weights...")

    # Setup directories
    os.makedirs("results", exist_ok=True)

    # Load data
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
        "student_direct_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth", MobileBPStudent)
    }

    # Add distilled models with different parameter combinations
    for f, s in zip(feature_loss_weights, soft_loss_weights):
        name = f"Distill(flw={f},slw={s})"
        path = f"student_best-distill_alpha=0.65-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight={f}-soft_loss_weight={s}-sp_weight=0.7-dp_weight=0.3.pth"
        model_paths[name] = (path, MobileBPStudent)

    # Load and evaluate models
    results_dict = {}
    for name, (path, model_class) in model_paths.items():
        full_path = os.path.join("models", path)
        model = load_model(model_class, full_path, config, device)
        if model:
            results = evaluate_model(model, X_test, y_test, device, name)
            results_dict[name] = results

    if len(results_dict) < len(model_paths):
        print("Some models failed to load - please ensure all required files exist")
        return

    # Generate individual error distribution plots
    for name, results in results_dict.items():
        plt.figure(figsize=(12, 5))

        # Systolic error distribution
        plt.subplot(1, 2, 1)
        sp_errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(sp_errors, bins=30, alpha=0.7, color='#2C7BB6')
        plt.xlabel('SP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - SP MAE Distribution')

        # Diastolic error distribution
        plt.subplot(1, 2, 2)
        dp_errors = np.abs(results['true_values'][:, 1] - results['predictions'][:, 1])
        plt.hist(dp_errors, bins=30, alpha=0.7, color='#D7191C')
        plt.xlabel('DP MAE (mmHg)')
        plt.ylabel('Frequency')
        plt.title(f'{name} - DP MAE Distribution')

        plt.tight_layout()
        plt.savefig(f'results/feature_soft_error_distribution_{name}.png', dpi=300)
        plt.close()

    # Generate comparison plots and tables
    plot_performance_comparison(results_dict, 'results/feature_soft_performance.png')
    create_comparison_table(results_dict)

    print("\nComparison completed! All results saved to 'results' directory")


if __name__ == "__main__":
    main()