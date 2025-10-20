# teacher_model.py 改进版
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ChannelAttention(nn.Module):
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
        b, _ = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, -1)
        y = self.fc(y)
        return x * y


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, expansion=4):
        super().__init__()
        hidden_dim = in_dim * expansion
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            ChannelAttention(in_dim)
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class BloodPressureTeacher(nn.Module):
    def __init__(self, config):
        super(BloodPressureTeacher, self).__init__()
        self.config = config
        
        # 加载预训练的ResNet34
        base_model = resnet34(pretrained=True)
        
        # 修改第一层以适应输入维度
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 从预训练模型复制其他层
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # 添加自定义层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, config.output_dim)
        
        # 用于存储中间层输出的钩子
        self.layer_outputs = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子以获取中间层输出"""
        def get_hook(name):
            def hook(module, input, output):
                self.layer_outputs[name] = output
            return hook
        
        # 为每个目标层注册钩子
        self.layer1.register_forward_hook(get_hook('layer1'))
        self.layer2.register_forward_hook(get_hook('layer2'))
        self.layer3.register_forward_hook(get_hook('layer3'))
    
    def forward(self, x):
        # 清除之前的中间层输出
        self.layer_outputs.clear()
        
        # 调整输入形状
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        x = x.unsqueeze(1)      # 添加通道维度
        
        # 通过ResNet各层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # 将产生layer1的输出
        x = self.layer2(x)  # 将产生layer2的输出
        x = self.layer3(x)  # 将产生layer3的输出
        x = self.layer4(x)
        
        # 全局池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout和全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        # 如果在训练模式下，返回预测值和中间层特征
        if self.training:
            return x, self.layer_outputs
        # 在评估模式下，只返回预测值
        return x
    
    def get_layer_output(self, layer_name):
        """获取指定层的输出"""
        return self.layer_outputs.get(layer_name, None)
