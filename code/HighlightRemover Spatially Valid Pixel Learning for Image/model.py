import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv+ELU 模块
class ConvELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ConvELU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(self.conv(x))

# Gated Convolution
class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.conv(x) * self.sigmoid(x)

# LFTModule 模块
class LFTModule(nn.Module):
    def __init__(self, in_channels):
        super(LFTModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)  # 融合后的卷积
        self.sigmoid = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        # 全局池化
        avg_pooled = self.avg_pool(x)  # (B, C, 1, 1)
        max_pooled = self.max_pool(x)  # (B, C, 1, 1)

        # 组合池化结果
        combined = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, 2C, 1, 1)

        # 生成显著性图 A
        A = self.sigmoid(self.conv(combined))  # (B, 1, H, W)

        # 有效像素区域归一化
        valid_mask = (A > 0.5).float()
        normalized_features = x * valid_mask  # (B, C, H, W)

        # 学习仿射参数
        gamma = self.conv(A)  # (B, C, H, W)
        beta = self.conv(A)   # (B, C, H, W)

        # 特征加权与输出
        F_out = normalized_features * gamma + beta
        return F_out

# Residual Convolution 模块
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

# Pyramid Pooling 模块
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = [x]
        for stage in self.stages:
            pyramid_features.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(pyramid_features, dim=1))

# CFBlock 模块
class CFBlock(nn.Module):
    def __init__(self, channels):
        super(CFBlock, self).__init__()
        self.lstm_h = nn.LSTM(channels, channels, batch_first=True, bidirectional=True)
        self.lstm_w = nn.LSTM(channels, channels, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 水平特征处理
        h_features = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        h_features, _ = self.lstm_h(h_features)
        h_features = h_features.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # 垂直特征处理
        w_features = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        w_features, _ = self.lstm_w(w_features)
        w_features = w_features.reshape(B, W, H, -1).permute(0, 3, 2, 1)
        
        # 特征融合生成 A2
        fused_features = torch.cat([h_features, w_features], dim=1)
        A2 = self.conv1(fused_features)
        
        # A3 和 A1 的智能逐元素乘法
        A3 = self.conv2(A2)
        A_out = A3 * x  # 智能逐元素乘法
        
        return A_out

# Bottleneck Module
class BottleneckModule(nn.Module):
    def __init__(self, channels):
        super(BottleneckModule, self).__init__()
        self.gated_conv1 = GatedConv(channels, channels)
        self.dilated_conv1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.cfblock1 = CFBlock(channels)

        self.gated_conv2 = GatedConv(channels, channels)
        self.dilated_conv2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.cfblock2 = CFBlock(channels)

        self.gated_conv3 = GatedConv(channels, channels)
        self.dilated_conv3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.cfblock3 = CFBlock(channels)

    def forward(self, x):
        A1 = self.cfblock1(self.lrelu(self.dilated_conv1(self.gated_conv1(x)))) + x
        A2 = self.cfblock2(self.lrelu(self.dilated_conv2(self.gated_conv2(A1)))) + A1
        A3 = self.cfblock3(self.lrelu(self.dilated_conv3(self.gated_conv3(A2)))) + A2
        return A3

# Highlight Removal Network
class HighlightRemovalNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(HighlightRemovalNet, self).__init__()
        # Encoder
        self.conv1 = ConvELU(in_channels, base_channels)
        self.conv2 = ConvELU(base_channels, base_channels * 2, dilation=2)
        self.conv3 = ConvELU(base_channels * 2, base_channels * 4, dilation=4)

        # Bottleneck
        self.bottleneck = BottleneckModule(base_channels * 4)

        # Decoder
        self.lft1 = LFTModule(base_channels * 4)
        self.conv4 = ConvELU(base_channels * 4, base_channels * 2)
        self.lft2 = LFTModule(base_channels * 2)
        self.conv5 = ConvELU(base_channels * 2, base_channels)
        self.gated_conv = GatedConv(base_channels, in_channels)

        # Final output layers
        self.residual_conv = ResidualConv(base_channels, base_channels)
        self.pyramid_pooling = PyramidPooling(base_channels, base_channels)
        self.conv_final = ConvELU(base_channels, in_channels)

    def forward(self, x):
        # Encoder
        F1 = self.conv1(x)
        F2 = self.conv2(F1)
        F3 = self.conv3(F2)

        # Bottleneck
        F4 = self.bottleneck(F3)

        # Decoder
        F5 = self.lft1(F4) + F2
        F6 = self.lft2(self.conv4(F5)) + F1
        F7 = self.gated_conv(self.conv5(F6))

        # Final output
        residual_features = self.residual_conv(F7 + x)
        pooled_features = self.pyramid_pooling(residual_features)
        output = self.conv_final(pooled_features)
        return output
