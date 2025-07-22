import torch
import os


class Config:
    # Data parameters
    input_dim = None
    output_dim = 2  # Two outputs: systolic and diastolic blood pressure

    max_features = 80
    MAX_FEATURES = 100
    SP_power = 0.75

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training hyperparameters
    batch_size = 32
    epochs = 200
    patience = 80  # Early stopping patience
    max_lr = 1e-4  # Maximum learning rate
    weight_decay = 1e-3  # L2 regularization

    # Loss function parameters
    huber_delta = 1.5  # Delta parameter for Huber loss
    sp_weight = 0.7  # Systolic blood pressure loss weight (increased)
    dp_weight = 0.3  # Diastolic blood pressure loss weight

    # Knowledge distillation parameters
    distill_alpha = 0.65  # Supervised learning weight
    distill_temp = 2.0  # Temperature scaling
    feature_loss_weight = 0.2  # Feature distillation weight
    soft_loss_weight = 0.8  # Soft target weight

    # Multi-layer feature distillation configuration
    feature_layers = {
        'early': {
            'teacher_layer': 'layer1',  # ResNet's first layer
            'student_layer': 'bneck1',  # MobileNetV3's first bottleneck
            'weight': 0.2
        },
        'middle': {
            'teacher_layer': 'layer2',  # ResNet's second layer
            'student_layer': 'bneck4',  # MobileNetV3's middle bottleneck
            'weight': 0.3
        },
        'late': {
            'teacher_layer': 'layer3',  # ResNet's third layer
            'student_layer': 'bneck7',  # MobileNetV3's later bottleneck
            'weight': 0.5
        }
    }

    # Warmup parameters
    pct_start = 0.3  # Percentage of cycle spent increasing LR

    # Mobile optimization parameters - enhanced
    quantize = True  # Enable quantization
    prune_threshold = 0.015  # Pruning threshold (increased)
    quantize_dtype = 'qint8'  # Quantization data type

    # Model save paths (unused)
    model_dir = "models"
    teacher_path = os.path.join(model_dir, "teacher_best.pth")
    student_path = os.path.join(model_dir, "student_best.pth")
    onnx_path = os.path.join(model_dir, "student_mobile.onnx")

    def __init__(self):
        """Ensure model directory exists"""
        os.makedirs(self.model_dir, exist_ok=True)