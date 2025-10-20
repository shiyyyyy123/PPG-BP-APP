# student_model.py - MobileNetV3实现
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """挤压激励模块(SE模块) - 简化版本，适用于全连接层"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        squeeze_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, squeeze_channels)
        self.fc2 = nn.Linear(squeeze_channels, in_channels)
        self.in_channels = in_channels

    def forward(self, x):
        # 适用于全连接层的SE块实现
        # 输入x的形状为[batch_size, channels]
        
        # 全局信息： 每个通道的平均值
        scale = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        
        # 对全局信息进行压缩和激励
        # 首先将scale调整为[batch_size, in_channels]以匹配fc1的输入要求
        scale_fc = self.fc1(x)  # 直接使用x作为输入，而不是scale
        scale_fc = F.relu(scale_fc)
        scale_fc = self.fc2(scale_fc)
        scale_fc = torch.sigmoid(scale_fc)
        
        # 应用通道注意力
        return x * scale_fc


class InvertedResidual(nn.Module):
    """MobileNetV3的倒置残差块"""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        self.identity = stride == 1 and inp == oup

        # 线性激活或h-swish激活
        act = nn.Hardswish if use_hs else nn.ReLU

        # 为SE块创建更简单、更鲁棒的实现
        se_module = SqueezeExcite(hidden_dim) if use_se else nn.Identity()

        self.conv = nn.Sequential(
            # 扩展层
            nn.Linear(inp, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act(),
            # SE层
            se_module,
            # 输出层
            nn.Linear(hidden_dim, oup),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileBPStudent(nn.Module):
    """基于MobileNetV3-Small的学生模型"""
    def __init__(self, config):
        super().__init__()
        
        # 特征维度配置
        input_dim = config.input_dim
        last_channel = 768  # 保持特征维度与教师兼容
        
        # 初始特征提取
        self.features = nn.Sequential(
            nn.Linear(input_dim, 16),  
            nn.BatchNorm1d(16),
            nn.Hardswish()
        )
        
        # MobileNetV3 架构参数 - 适配全连接层情况
        # [扩展率, 输出通道, 使用SE, 使用HS, 步长]
        self.cfg = [
            # k, t, c, SE, HS, s 
            [3,  16,  16,  True,  False, 2],
            [3,  72,  24,  False, False, 2],
            [3,  88,  24,  False, False, 1],
            [5,  96,  40,  True,  True,  2],
            [5, 240,  40,  True,  True,  1],
            [5, 240,  40,  True,  True,  1],
            [5, 120,  48,  True,  True,  1],
            [5, 144,  48,  True,  True,  1],
            [5, 288,  96,  True,  True,  2],
            [5, 576,  96,  True,  True,  1],
            [5, 576,  96,  True,  True,  1],
        ]
        
        # 构建主干网络
        input_channel = 16
        self.blocks = nn.ModuleList()
        
        # 创建MobileNetV3层
        for k, t, c, use_se, use_hs, s in self.cfg:
            exp_size = t
            output_channel = c
            self.blocks.append(
                InvertedResidual(
                    input_channel, 
                    exp_size, 
                    output_channel, 
                    k, s, 
                    use_se, 
                    use_hs
                )
            )
            input_channel = output_channel
        
        # 最后特征提取层
        self.conv = nn.Sequential(
            nn.Linear(input_channel, last_channel),
            nn.BatchNorm1d(last_channel),
            nn.Hardswish()
        )
        
        # 特征降维以保持与教师模型兼容的特征维度
        self.feature_adapter = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish()
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, config.output_dim)
        )

    def extract_features(self, x):
        x = self.features(x)
        
        # 收集中间特征用于蒸馏
        distill_features = []
        
        # 保存feature1 - 初始特征
        distill_features.append(x)
        
        # 特定点收集特征
        feature_indices = [3, 7, 10]  # 根据模型结构选择关键点
        for i, block in enumerate(self.blocks):
            try:
                x = block(x)
                if i in feature_indices:
                    distill_features.append(x)
            except Exception as e:
                print(f"Block {i} 处理失败: {e}")
                print(f"输入形状: {x.shape}")
                raise
        
        # 最终特征
        x = self.conv(x)
        distill_features.append(x)
        
        # 适配器转换
        x = self.feature_adapter(x)
        distill_features.append(x)
        
        return x, distill_features

    def forward(self, x):
        x, features = self.extract_features(x)
        return self.output(x), features

