import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.models import resnet50

# Context Mixer (CM) Module
class ContextMixer(nn.Module):
    def __init__(self, in_channels):
        super(ContextMixer, self).__init__()
        # SCNN 分支: 四个不同核大小的 SCNN_D, SCNN_U, SCNN_R, SCNN_L
        self.scnn_d = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        self.scnn_u = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        self.scnn_r = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        self.scnn_l = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        self.scnn_deconv = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)

        # Res 分支
        self.res_branch = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)

        # SE 分支
        self.se_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # SCNN 分支: 分别处理后连接并反卷积
        scnn_d = self.scnn_d(x)
        scnn_u = self.scnn_u(x)
        scnn_r = self.scnn_r(x)
        scnn_l = self.scnn_l(x)
        scnn_combined = torch.cat([scnn_d, scnn_u, scnn_r, scnn_l], dim=1)
        scnn_output = self.scnn_deconv(scnn_combined)

        # Res 分支
        res = self.res_branch(x)

        # SE 分支
        se = self.se_branch(x)

        # 特征融合
        return scnn_output * se + res

# LSHDNet implementation
class LSHDNet(nn.Module):
    def __init__(self):
        super(LSHDNet, self).__init__()
        backbone = resnet50(pretrained=True)

        # 提取 ResNet50 各阶段的特征
        self.stage1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        # CM 模块用于处理各分辨率特征
        self.cm1 = ContextMixer(256)
        self.cm2 = ContextMixer(512)
        self.cm3 = ContextMixer(1024)
        self.cm4 = ContextMixer(2048)
        self.cm5 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 用于初始输入的卷积处理

        # 输出卷积层
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 提取多尺度特征
        feat0 = self.cm5(x)  # 初始输入处理，输出形状 (batch, 64, H, W)
        feat1 = self.stage1(x)  # 输出形状 (batch, 256, H/4, W/4)
        feat2 = self.stage2(feat1)  # 输出形状 (batch, 512, H/8, W/8)
        feat3 = self.stage3(feat2)  # 输出形状 (batch, 1024, H/16, W/16)
        feat4 = self.stage4(feat3)  # 输出形状 (batch, 2048, H/32, W/32)

        # CM 模块处理各特征
        cm0 = feat0  # 初始特征无需再次处理
        cm1 = self.cm1(feat1)
        cm2 = self.cm2(feat2)
        cm3 = self.cm3(feat3)
        cm4 = self.cm4(feat4)

        # 构建上下文特征金字塔
        combined = F.interpolate(cm4, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   F.interpolate(cm3, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   F.interpolate(cm2, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   F.interpolate(cm1, size=cm0.shape[2:], mode='bilinear', align_corners=False) + cm0

        # 输出最终的高光检测结果
        return self.output_conv(combined)

# Refinement Module (RM)
class RM(nn.Module):
    def __init__(self):
        super(RM, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x) + x
        return self.conv2(x)

# Attention Fusion Module (AFM)
class AFM(nn.Module):
    def __init__(self):
        super(AFM, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, f1, f2):
        combined = torch.cat([f1, f2], dim=1)
        attention = F.sigmoid(self.conv2(F.relu(self.conv1(combined))))
        weighted_f1 = attention[:, :1, :, :] * f1
        weighted_f2 = attention[:, 1:, :, :] * f2
        return weighted_f1 + weighted_f2

# Main detection pipeline
class HighlightDetectionPipeline:
    def __init__(self):
        self.lshdnet = LSHDNet()
        self.rm_modules = [RM() for _ in range(6)]
        self.afm = AFM()

    def forward(self, img):
        # Segment images into 256p, 512p, 1024p, 2048p
        scales = [256, 512, 1024, 2048]
        resized_images = [Resize((s, s))(img) for s in scales]
        
        # Process with LSHDNet
        features = [self.lshdnet(resized) for resized in resized_images]

        # Apply RMs
        f5 = self.rm_modules[0](features[3], features[2])
        f6 = self.rm_modules[1](f5, features[1])
        f7 = self.rm_modules[2](f6, features[0])
        
        f8 = self.rm_modules[3](features[1], features[0])
        f9 = self.rm_modules[4](f8, features[2])
        f10 = self.rm_modules[5](f9, features[3])

        # Apply AFM
        final_output = self.afm(f10, f7)
        return final_output

# Instantiate and process an example
pipeline = HighlightDetectionPipeline()
dummy_input = torch.randn(1, 3, 2048, 2048)  # Example input
output = pipeline.forward(dummy_input)
print("Final output shape:", output.shape)
