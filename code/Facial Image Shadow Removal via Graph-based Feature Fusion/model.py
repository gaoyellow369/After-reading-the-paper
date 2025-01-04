import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 1. 几何模型生成与光流计算
class PreprocessingModule:
    def __init__(self):
        pass

    def generate_geometric_model(self, img):
        """
        使用 Mediapipe 检测人脸关键点，标记 2D 关键点。
        """
        # 假设 Mediapipe 已集成 (代码略，返回关键点)
        h, w, _ = img.shape
        keypoints = np.array([[int(w * 0.3), int(h * 0.4)], [int(w * 0.7), int(h * 0.4)]])  # 模拟关键点
        return keypoints

    def flip_image(self, img, keypoints):
        """
        翻转图像并调整关键点位置。
        """
        flipped_img = cv2.flip(img, 1)
        h, w, _ = img.shape
        flipped_keypoints = keypoints.copy()
        flipped_keypoints[:, 0] = w - flipped_keypoints[:, 0]
        return flipped_img, flipped_keypoints

    def calculate_optical_flow(self, img1, img2):
        """
        计算光流，描述像素从翻转前到翻转后的运动信息。
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def transfer_illumination(self, img, flow):
        """
        利用光流信息调整阴影区域，生成粗略少阴影图像。
        """
        h, w, _ = img.shape
        result = img.copy()
        for y in range(h):
            for x in range(w):
                dx, dy = flow[y, x]
                src_x, src_y = int(x + dx), int(y + dy)
                if 0 <= src_x < w and 0 <= src_y < h:
                    result[y, x] = img[src_y, src_x]
        return result

    def generate_shadowless_image(self, img):
        """
        生成粗略的少阴影图像。
        """
        keypoints = self.generate_geometric_model(img)
        flipped_img, flipped_keypoints = self.flip_image(img, keypoints)
        flow = self.calculate_optical_flow(img, flipped_img)
        shadowless_img = self.transfer_illumination(img, flow)
        return shadowless_img

# 2. MultiScaleDivider
class MultiScaleDivider(nn.Module):
    def __init__(self):
        super(MultiScaleDivider, self).__init__()

    def forward(self, x):
        """
        输入: 图像张量 (B, C, H, W)
        输出: 四个不同尺度的特征 [scale1, scale2, scale3, scale4]
        """
        scale1 = F.interpolate(x, scale_factor=1.0, mode='bilinear', align_corners=False)
        scale2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        scale3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        scale4 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        return [scale1, scale2, scale3, scale4]

# 3. MEncoder
class MEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.final_conv = nn.Conv2d(output_dim * 5, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))

        pool1 = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        pool2 = F.adaptive_avg_pool2d(x, (x.size(2) // 4, x.size(3) // 4))
        pool3 = F.adaptive_avg_pool2d(x, (x.size(2) // 8, x.size(3) // 8))
        pool4 = F.adaptive_avg_pool2d(x, (1, 1))

        pool1 = self.relu(self.conv1(pool1))
        pool2 = self.relu(self.conv1(pool2))
        pool3 = self.relu(self.conv1(pool3))
        pool4 = self.relu(self.conv1(pool4))

        fused = torch.cat([x, pool1, pool2, pool3, pool4], dim=1)
        output = self.relu(self.final_conv(fused))
        return output

# 4. GCEncoder
class GraphProcessingModule(nn.Module):
    def __init__(self, input_dim, output_dim, k=8):
        super(GraphProcessingModule, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.batch_norm1 = nn.BatchNorm1d(output_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.drop_path = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = self.gelu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        x = self.drop_path(x)
        return x + residual

class FeedForwardModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardModule, self).__init__()
        self.fc1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.drop_path = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.drop_path(self.fc2(x))
        return x + residual

class GCEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, k=8):
        super(GCEncoder, self).__init__()
        self.graph_processing = GraphProcessingModule(input_dim, output_dim, k)
        self.feed_forward = FeedForwardModule(output_dim, output_dim * 2)

    def forward(self, x):
        graph_feature = self.graph_processing(x)
        output = self.feed_forward(graph_feature)
        return output

# 5. Feature Modulation
class FeatureModulation(nn.Module):
    def __init__(self, input_dim):
        super(FeatureModulation, self).__init__()
        self.gamma_conv = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.beta_conv = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.instance_norm = nn.InstanceNorm2d(input_dim)

    def forward(self, fm_encoder, fg_encoder):
        gamma = self.gamma_conv(fg_encoder)
        beta = self.beta_conv(fg_encoder)
        fm_encoder = self.instance_norm(fm_encoder)
        return fm_encoder * gamma + beta

# 6. Fusion Decoder
class FusionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusionDecoder, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, scale4, fm1, fm2, fm3, fm_mod1, fm_mod2, fm_mod3):
        """
        输入:
        - scale4: 第四尺度特征 (B, C, H/8, W/8)
        - fm1, fm2, fm3: 原始图像的多尺度特征
        - fm_mod1, fm_mod2, fm_mod3: 调制后的多尺度特征
        输出:
        - 最终生成的图像 (B, 3, H, W)
        """
        # 第一步
        x = F.interpolate(scale4, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样到 H/4
        x = torch.cat([x, fm1, fm_mod1], dim=1)  # 与 FM1 和 FMOD1 拼接

        # 第二步
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样到 H/2
        x = torch.cat([x, fm2, fm_mod2], dim=1)  # 与 FM2 和 FMOD2 拼接

        # 第三步
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样到 H
        x = torch.cat([x, fm3, fm_mod3], dim=1)  # 与 FM3 和 FMOD3 拼接

        # 最终卷积生成输出
        x = self.relu(self.conv(x))
        return x


# 测试代码
if __name__ == "__main__":
    B, C, H, W = 4, 3, 256, 256
    input_image = torch.randn(B, C, H, W)

    # 生成少阴影图像
    preprocessor = PreprocessingModule()
    shadowless_image = preprocessor.generate_shadowless_image(input_image.numpy().transpose(0, 2, 3, 1)[0]).transpose(2, 0, 1)[None, ...]

    # Multi-scale division
    divider = MultiScaleDivider()
    original_scales = divider(input_image)
    shadowless_scales = divider(torch.tensor(shadowless_image))

    # MEncoder and GCEncoder
    mencoder = MEncoder(input_dim=3, output_dim=64)
    gcencoder = GCEncoder(input_dim=64, output_dim=64)
    feature_modulation = FeatureModulation(input_dim=64)
    fusion_decoder = FusionDecoder(input_dim=192, output_dim=3)

    fm_encoders = [mencoder(scale) for scale in original_scales[:3]]
    fg_encoders = [gcencoder(fm.permute(0, 2, 3, 1).flatten(2).permute(0, 2, 1)) for fm in shadowless_scales[:3]]
    fg_encoders = [fg.view(B, 64, scale.size(2), scale.size(3)) for fg, scale in zip(fg_encoders, original_scales[:3])]

    fm_modulations = [feature_modulation(fm, fg) for fm, fg in zip(fm_encoders, fg_encoders)]

    output = fusion_decoder(original_scales[3], fm_encoders[0], fm_encoders[1], fm_encoders[2],
                            fm_modulations[0], fm_modulations[1], fm_modulations[2])

    print("Output shape:", output.shape)
