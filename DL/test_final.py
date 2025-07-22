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

    model = model_class(config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(config.device)
        start_time = time.time()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            preds = outputs[0].cpu().numpy()
        else:
            preds = outputs.cpu().numpy()
        inference_time = (time.time() - start_time) * 1000  # 毫秒
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
    plt.close()


def main():
    print("=" * 50)
    print("Blood Pressure Prediction Model Final Testing Script")
    print("=" * 50)

    os.makedirs("results", exist_ok=True)
    config = Config()


    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    print(f"Test set size: {X_test.shape}")
    config.input_dim = X_test.shape[1]

    device = config.device
    print(f"Using device: {device}")


    teacher_path = "models/teacher_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    student_direct_path = "models/student_direct_best-epochs=200-batch_size=32-lr=0.0001-huber_delta=1.5-sp_weight=0.7.pth"
    student_distill_path = "models/student_best-distill_alpha=0.65-epochs=200-batch_size=32-lr=0.0001-distill_temp=2.0-huber_delta=1.5-feature_loss_weight=0.3-soft_loss_weight=0.7-sp_weight=0.7-dp_weight=0.3.pth"


    teacher_results = None
    if os.path.exists(teacher_path):
        teacher_results = load_and_evaluate_model(
            teacher_path, BloodPressureTeacher, X_test, y_test, config, "Teacher Model (ResNet)")
        plot_predictions(teacher_results, "Teacher Model (ResNet) Prediction Results",
                         "results/teacher_predictions.png")
    else:
        print("Teacher model weights not found! Please train the teacher model first.")


    student_direct_results = None
    if os.path.exists(student_direct_path):
        student_direct_results = load_and_evaluate_model(
            student_direct_path, MobileBPStudent, X_test, y_test, config, "Student Model (Direct)")
        plot_predictions(student_direct_results, "Student Model (Direct) Prediction Results",
                         "results/student_direct_predictions.png")
    else:
        print("Direct-trained student model weights not found! Please train the student model first.")


    student_distill_results = None
    if os.path.exists(student_distill_path):
        student_distill_results = load_and_evaluate_model(
            student_distill_path, MobileBPStudent, X_test, y_test, config, "Student Model (Distill)")
        plot_predictions(student_distill_results, "Student Model (Distill) Prediction Results",
                         "results/student_distill_predictions.png")
    else:
        print("Distilled student model weights not found! Please train the distilled student model first.")


    print("\nModel Performance Comparison:")
    results_list = [r for r in [teacher_results, student_direct_results, student_distill_results] if r is not None]

    if results_list:
        import pandas as pd
        df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'MAE': r['mae'],
                'Systolic_MAE': r['sp_mae'],
                'Diastolic_MAE': r['dp_mae'],
                'RMSE': r['rmse'],
                'R2': r['r2'],
                'Inference_Time(ms)': r['inference_time'],
                'Model_Size(MB)': os.path.getsize(teacher_path if r['model_name'].startswith('Teacher') else
                                                  (student_direct_path if r['model_name'].endswith(
                                                      'Direct)') else student_distill_path)) / (1024 * 1024)
            }
            for r in results_list
        ])
        print(df.to_string(index=False))
        df.to_csv('results/final_model_comparison.csv', index=False, encoding='utf-8-sig')

    print("\nTesting completed! All results saved to 'results' directory")


if __name__ == "__main__":
    main()