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
    """Load model weights from checkpoint"""
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

        # Time inference
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


def plot_error_distribution(teacher_results, student_results, save_path):
    """Plot comparison of error distributions"""
    plt.figure(figsize=(15, 6))

    # Systolic error distribution
    plt.subplot(1, 2, 1)
    teacher_sp_errors = np.abs(teacher_results['true_values'][:, 0] - teacher_results['predictions'][:, 0])
    student_sp_errors = np.abs(student_results['true_values'][:, 0] - student_results['predictions'][:, 0])

    plt.hist(teacher_sp_errors, bins=30, alpha=0.5, label=f'Teacher Model (MAE={teacher_results["sp_mae"]:.2f})')
    plt.hist(student_sp_errors, bins=30, alpha=0.5, label=f'Student Model (MAE={student_results["sp_mae"]:.2f})')
    plt.xlabel('Systolic Pressure Absolute Error (mmHg)')
    plt.ylabel('Frequency')
    plt.title('Systolic Pressure Error Distribution')
    plt.legend()

    # Diastolic error distribution
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
    """Create performance comparison bar chart"""
    metrics = ['mae', 'sp_mae', 'dp_mae', 'high_sp_mae']
    metric_labels = ['Overall MAE', 'Systolic MAE', 'Diastolic MAE', 'High BP MAE']

    teacher_values = [teacher_results[m] for m in metrics]
    student_values = [student_results[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, teacher_values, width, label='Teacher Model', color='#2C7BB6')
    plt.bar(x + width / 2, student_values, width, label='Student Model', color='#D7191C')

    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Error (mmHg)')
    plt.title('Teacher vs Student Model Performance')
    plt.xticks(x, metric_labels)
    plt.legend()

    # Add value labels
    for i, v in enumerate(teacher_values):
        plt.text(i - width / 2, v + 0.1, f'{v:.2f}', ha='center')
    for i, v in enumerate(student_values):
        plt.text(i + width / 2, v + 0.1, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_comparison_table(teacher_results, student_results):
    """Generate performance comparison table"""
    data = {
        'Model': ['Teacher Model', 'Student Model'],
        'Overall MAE': [teacher_results['mae'], student_results['mae']],
        'Systolic MAE': [teacher_results['sp_mae'], student_results['sp_mae']],
        'Diastolic MAE': [teacher_results['dp_mae'], student_results['dp_mae']],
        'High BP MAE': [teacher_results['high_sp_mae'], student_results['high_sp_mae']],
        'Inference Time (ms/sample)': [teacher_results['inference_time'], student_results['inference_time']]
    }

    df = pd.DataFrame(data)
    print("\nModel Performance Comparison:")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('results/teacher_student_comparison.csv', index=False, encoding='utf-8-sig')
    return df


def main():
    """Main function: Compare teacher and distilled student model performance"""
    print("Starting teacher vs distilled student model comparison...")

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load config and data
    config = Config()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    config.input_dim = X_test.shape[1]

    device = config.device
    print(f"Using device: {device}")

    # Load models
    teacher_model = load_model(BloodPressureTeacher, "models/teacher_best.pth", config, device)
    student_model = load_model(MobileBPStudent, "models/student_best.pth", config, device)

    if teacher_model is None or student_model is None:
        print("Model loading failed - please ensure model files exist")
        return

    # Evaluate models
    teacher_results = evaluate_model(teacher_model, X_test, y_test, device, "Teacher Model")
    student_results = evaluate_model(student_model, X_test, y_test, device, "Student Model")

    # Generate visual comparisons
    plot_error_distribution(teacher_results, student_results, 'results/error_distribution_comparison.png')
    plot_performance_comparison(teacher_results, student_results, 'results/performance_comparison.png')

    # Create comparison table
    comparison_df = create_comparison_table(teacher_results, student_results)

    # Calculate model sizes
    teacher_size = os.path.getsize("models/teacher_best.pth") / (1024 * 1024)  # MB
    student_size = os.path.getsize("models/student_best.pth") / (1024 * 1024)  # MB

    # Print summary report
    print("\nPerformance Comparison Summary:")
    print(f"Model Size Comparison:")
    print(f"- Teacher Model: {teacher_size:.2f} MB")
    print(f"- Student Model: {student_size:.2f} MB")
    print(f"- Compression Ratio: {teacher_size / student_size:.2f}x")

    print(f"\nPrediction Performance:")
    print(f"- Overall MAE Difference: {student_results['mae'] - teacher_results['mae']:.2f} mmHg")
    print(f"- Systolic MAE Difference: {student_results['sp_mae'] - teacher_results['sp_mae']:.2f} mmHg")
    print(f"- Diastolic MAE Difference: {student_results['dp_mae'] - teacher_results['dp_mae']:.2f} mmHg")

    print(f"\nInference Speed Comparison:")
    speedup = teacher_results['inference_time'] / student_results['inference_time']
    print(f"- Speed Improvement: {speedup:.2f}x")

    print("\nComparison completed! All results saved to 'results' directory")


if __name__ == "__main__":
    main()