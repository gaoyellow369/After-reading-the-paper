import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
from model import PreprocessingModule, MultiScaleDivider, MEncoder, GCEncoder, FeatureModulation, FusionDecoder

# 数据集定义
class ShadowRemovalDataset(Dataset):
    def __init__(self, shadow_dir, shadow_free_dir, transform=None):
        self.shadow_dir = shadow_dir
        self.shadow_free_dir = shadow_free_dir
        self.shadow_images = sorted(os.listdir(shadow_dir))
        self.shadow_free_images = sorted(os.listdir(shadow_free_dir))
        self.transform = transform

    def __len__(self):
        return len(self.shadow_images)

    def __getitem__(self, idx):
        shadow_path = os.path.join(self.shadow_dir, self.shadow_images[idx])
        shadow_free_path = os.path.join(self.shadow_free_dir, self.shadow_free_images[idx])

        shadow_image = Image.open(shadow_path).convert("RGB")
        shadow_free_image = Image.open(shadow_free_path).convert("RGB")

        if self.transform:
            shadow_image = self.transform(shadow_image)
            shadow_free_image = self.transform(shadow_free_image)

        return shadow_image, shadow_free_image

# 感知损失定义
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.layers = [vgg[:4], vgg[:9], vgg[:18], vgg[:27], vgg[:36]]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        loss = 0
        for layer in self.layers:
            pred = layer(pred)
            target = layer(target)
            loss += nn.functional.mse_loss(pred, target)
        return loss

# 总损失定义
class ShadowRemovalLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ShadowRemovalLoss, self).__init__()
        self.visual_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        l_visual = self.visual_loss(pred, target)
        l_perceptual = self.perceptual_loss(pred, target)
        return l_visual + self.alpha * l_perceptual

# 主模型定义
class ShadowRemovalModel(nn.Module):
    def __init__(self):
        super(ShadowRemovalModel, self).__init__()
        self.divider = MultiScaleDivider()
        self.mencoder = MEncoder(input_dim=3, output_dim=64)
        self.gcencoder = GCEncoder(input_dim=64, output_dim=64)
        self.feature_modulation = FeatureModulation(input_dim=64)
        self.fusion_decoder = FusionDecoder(input_dim=192, output_dim=3)

    def forward(self, original_image, shadowless_image):
        # 多尺度划分
        original_scales = self.divider(original_image)
        shadowless_scales = self.divider(shadowless_image)

        # MEncoder 和 GCEncoder
        fm_encoders = [self.mencoder(scale) for scale in original_scales[:3]]
        fg_encoders = [
            self.gcencoder(fm.permute(0, 2, 3, 1).flatten(2).permute(0, 2, 1))
            for fm in shadowless_scales[:3]
        ]
        fg_encoders = [
            fg.view(original_image.size(0), 64, scale.size(2), scale.size(3))
            for fg, scale in zip(fg_encoders, original_scales[:3])
        ]

        # 特征调制
        fm_modulations = [
            self.feature_modulation(fm, fg) for fm, fg in zip(fm_encoders, fg_encoders)
        ]

        # 融合解码
        output = self.fusion_decoder(
            original_scales[3],
            fm_encoders[0],
            fm_encoders[1],
            fm_encoders[2],
            fm_modulations[0],
            fm_modulations[1],
            fm_modulations[2],
        )
        return output

# 训练流程
def train_model(model, dataset, num_epochs=20, learning_rate=1e-4, alpha=0.5):
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = ShadowRemovalLoss(alpha)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for shadow_image, shadow_free_image in train_loader:
            shadow_image = shadow_image.to(device)
            shadow_free_image = shadow_free_image.to(device)

            # 生成少阴影图像
            shadowless_image = shadow_image.clone().cpu().numpy()
            shadowless_image = torch.tensor(
                shadowless_image, dtype=torch.float32, device=device
            )

            # 前向传播
            optimizer.zero_grad()
            pred_image = model(shadow_image, shadowless_image)

            # 计算损失
            loss = loss_fn(pred_image, shadow_free_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# 数据加载和训练
if __name__ == "__main__":
    transform = ToTensor()
    dataset = ShadowRemovalDataset(
        shadow_dir="shadow_images/", shadow_free_dir="shadow_free_images/", transform=transform
    )
    model = ShadowRemovalModel()
    train_model(model, dataset)
