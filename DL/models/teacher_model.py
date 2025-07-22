import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ChannelAttention(nn.Module):
    """Channel Attention Module for 1D features"""

    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel-wise attention weights"""
        b, _ = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, -1)  # Global average pooling
        y = self.fc(y)  # Attention weights
        return x * y  # Apply attention


class ResNetBlock(nn.Module):
    """Custom ResNet-inspired block with channel attention"""

    def __init__(self, in_dim, expansion=4):
        super().__init__()
        hidden_dim = in_dim * expansion
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            ChannelAttention(in_dim)  # Add channel attention
        )

    def forward(self, x):
        """Residual connection with GELU activation"""
        return F.gelu(x + self.block(x))


class BloodPressureTeacher(nn.Module):
    """Teacher model based on ResNet34 architecture for blood pressure estimation

    Features:
    - Modified ResNet34 backbone pretrained on ImageNet
    - Custom input adaptation for 1D signals
    - Intermediate feature extraction for knowledge distillation
    - Channel attention mechanisms
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load pretrained ResNet34
        base_model = resnet34(pretrained=True)

        # Modify first layer for 1D signal input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Transfer layers from pretrained model
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Custom classification head
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, config.output_dim)

        # Feature extraction setup
        self.layer_outputs = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture intermediate layer outputs"""

        def get_hook(name):
            def hook(module, input, output):
                self.layer_outputs[name] = output

            return hook

        # Register hooks on key layers
        self.layer1.register_forward_hook(get_hook('layer1'))
        self.layer2.register_forward_hook(get_hook('layer2'))
        self.layer3.register_forward_hook(get_hook('layer3'))

    def forward(self, x):
        """Forward pass with feature extraction

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            If training: tuple of (predictions, intermediate_features)
            If eval: predictions only
        """
        # Clear previous features
        self.layer_outputs.clear()

        # Reshape 1D input to 2D format
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = x.unsqueeze(1)  # Add dummy spatial dimension

        # Forward through ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        # Return features during training for distillation
        if self.training:
            return x, self.layer_outputs
        return x

    def get_layer_output(self, layer_name):
        """Retrieve intermediate features by layer name

        Args:
            layer_name: One of ['layer1', 'layer2', 'layer3']

        Returns:
            Tensor containing the requested layer's output
            None if layer name is invalid
        """
        return self.layer_outputs.get(layer_name, None)