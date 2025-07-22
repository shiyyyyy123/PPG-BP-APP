import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Simplified Squeeze-and-Excitation block for fully connected layers"""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeeze_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, squeeze_channels)
        self.fc2 = nn.Linear(squeeze_channels, in_channels)
        self.in_channels = in_channels

    def forward(self, x):
        """Forward pass with channel-wise attention mechanism

        Args:
            x: Input tensor of shape [batch_size, channels]

        Returns:
            Output tensor with channel-wise attention weights applied
        """
        # Simplified SE implementation for FC layers
        scale_fc = self.fc1(x)  # Process entire input for attention
        scale_fc = F.relu(scale_fc)
        scale_fc = self.fc2(scale_fc)
        scale_fc = torch.sigmoid(scale_fc)

        return x * scale_fc  # Apply channel attention weights


class InvertedResidual(nn.Module):
    """MobileNetV3-style inverted residual block adapted for FC layers"""

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        self.identity = stride == 1 and inp == oup

        # Activation selection
        act = nn.Hardswish if use_hs else nn.ReLU

        # SE module configuration
        se_module = SqueezeExcite(hidden_dim) if use_se else nn.Identity()

        self.conv = nn.Sequential(
            # Expansion layer
            nn.Linear(inp, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act(),

            # Channel attention
            se_module,

            # Projection layer
            nn.Linear(hidden_dim, oup),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        """Forward pass with optional residual connection"""
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


class MobileBPStudent(nn.Module):
    """Lightweight student model based on MobileNetV3 architecture for blood pressure estimation

    Key Features:
    - Adapted MobileNetV3 architecture for fully connected layers
    - Simplified Squeeze-and-Excitation blocks
    - Feature extraction points for knowledge distillation
    - Compatible feature dimensions with teacher model
    """

    def __init__(self, config):
        super().__init__()

        # Network configuration
        input_dim = config.input_dim
        last_channel = 768  # Maintains compatibility with teacher features

        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Hardswish()
        )

        # MobileNetV3 architecture parameters adapted for FC layers
        # [expansion, output_channels, use_se, use_hs, stride]
        self.cfg = [
            [3, 16, 16, True, False, 2],  # Block 0
            [3, 72, 24, False, False, 2],  # Block 1
            [3, 88, 24, False, False, 1],  # Block 2
            [5, 96, 40, True, True, 2],  # Block 3
            [5, 240, 40, True, True, 1],  # Block 4
            [5, 240, 40, True, True, 1],  # Block 5
            [5, 120, 48, True, True, 1],  # Block 6
            [5, 144, 48, True, True, 1],  # Block 7
            [5, 288, 96, True, True, 2],  # Block 8
            [5, 576, 96, True, True, 1],  # Block 9
            [5, 576, 96, True, True, 1],  # Block 10
        ]

        # Build backbone network
        input_channel = 16
        self.blocks = nn.ModuleList()

        for k, t, c, use_se, use_hs, s in self.cfg:
            self.blocks.append(
                InvertedResidual(
                    input_channel,
                    t,  # Expansion size
                    c,  # Output channels
                    k,  # Kernel size (unused in FC)
                    s,  # Stride
                    use_se,
                    use_hs
                )
            )
            input_channel = c

        # Final feature processing
        self.conv = nn.Sequential(
            nn.Linear(input_channel, last_channel),
            nn.BatchNorm1d(last_channel),
            nn.Hardswish()
        )

        # Feature adapter for teacher compatibility
        self.feature_adapter = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish()
        )

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, config.output_dim)
        )

    def extract_features(self, x):
        """Feature extraction with distillation points

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            tuple: (main_features, list_of_distillation_features)
        """
        x = self.features(x)

        # Initialize feature collection for distillation
        distill_features = [x]  # Initial features

        # Strategic points for feature collection
        feature_indices = [3, 7, 10]  # Key blocks for distillation

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_indices:
                distill_features.append(x)

        # Final processing
        x = self.conv(x)
        distill_features.append(x)

        # Feature adaptation
        x = self.feature_adapter(x)
        distill_features.append(x)

        return x, distill_features

    def forward(self, x):
        """Forward pass with feature extraction

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            tuple: (predictions, list_of_features_for_distillation)
        """
        x, features = self.extract_features(x)
        return self.output(x), features