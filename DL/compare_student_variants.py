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
    """Load model from checkpoint"""
    model = MobileBPStudent(config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"Loading failed: {e}")
        return None


def evaluate_model(model, X, y, device, model_name="Model"):
    """Evaluate model performance metrics"""
    model.eval()
    results = {}

    with torch.no_grad():
        inputs = torch.FloatTensor(X).to(device)

        # Start timer
        start_time = time.time()
        outputs = model(inputs)

        # Extract predictions if output is tuple (predictions, features)
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        per_sample_time = inference_time / len(X)

    # Calculate evaluation metrics
    sp_mae = mean_absolute_error(y[:, 0], preds[:, 0])  # Systolic
    dp_mae = mean_absolute_error(y[:, 1], preds[:, 1])  # Diastolic
    mae = mean_absolute_error(y, preds)  # Overall

    # Calculate errors for different blood pressure ranges
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
    plt.figure(figsize=(15, 6))
    colors = ['#2C7BB6', '#D7191C']  # Blue, Red

    # Systolic error distribution
    plt.subplot(1, 2, 1)
    for i, (name, results) in enumerate(results_dict.items()):
        errors = np.abs(results['true_values'][:, 0] - results['predictions'][:, 0])
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                 label=f'{name} (MAE={results["sp_mae"]:.2f})')

    plt.xlabel('Systolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Systolic Pressure Error Distribution')
    plt.legend()

    # Diastolic error distribution
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
    """Create performance comparison bar chart"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High BP MAE']

    n_models = len(results_dict)
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 6))
    colors = ['#2C7BB6', '#D7191C']  # Blue, Red

    for i, (name, results) in enumerate(results_dict.items()):
        values = [results[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = plt.bar(x + offset, values, width, label=name, color=colors[i])

        # Add value labels
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

    # Save to CSV
    df.to_csv('results/student_variants_comparison.csv', index=False, encoding='utf-8-sig')
    return df


def main():
    """Main function: Compare different student model variants"""
    print("Starting comparison of student model variants...")

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load config and data
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]

    device = config.device
    print(f"Using device: {device}")

    # Model paths
    model_paths = {
        "Distilled Student": "models/student_best-distill_alpha=0.65-epochs=300-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.8-soft_loss_weight=0.2-sp_weight=0.7-dp_weight=0.3.pth",
        "Direct Trained Student": "models/student_direct_best-epochs=300-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    }

    # Load and evaluate models
    results_dict = {}
    for name, model_path in model_paths.items():
        model = load_model(model_path, config, device)
        if model:
            results = evaluate_model(model, X_test, y_test, device, name)
            results_dict[name] = results

    if len(results_dict) < 2:
        print("Model loading failed - please ensure all required model files exist")
        return

    # Generate visual comparisons
    plot_error_distribution(results_dict, 'results/student_variants_error_distribution.png')
    plot_performance_comparison(results_dict, 'results/student_variants_performance.png')

    # Create comparison table
    comparison_df = create_comparison_table(results_dict)

    # Calculate and print improvement percentages
    base_model = results_dict["Direct Trained Student"]
    distill_model = results_dict["Distilled Student"]

    print("\nKnowledge Distillation Improvement Analysis:")
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_names = ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High BP MAE']

    for metric, name in zip(metrics, metric_names):
        improvement = (base_model[metric] - distill_model[metric]) / base_model[metric] * 100
        print(f"{name} improvement: {improvement:.2f}%")

    print("\nComparison completed! All results saved to 'results' directory")


if __name__ == "__main__":
    main()