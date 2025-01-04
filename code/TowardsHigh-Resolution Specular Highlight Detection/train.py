import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from torchvision.models import resnet50
from tqdm import tqdm

# Context Mixer (CM) Module
class ContextMixer(nn.Module):
    def __init__(self, in_channels):
        super(ContextMixer, self).__init__()
        self.scnn_d = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        self.scnn_u = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        self.scnn_r = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        self.scnn_l = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 3), padding=(0, 1))
        self.scnn_deconv = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.res_branch = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.se_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scnn_d = self.scnn_d(x)
        scnn_u = self.scnn_u(x)
        scnn_r = self.scnn_r(x)
        scnn_l = self.scnn_l(x)
        scnn_combined = torch.cat([scnn_d, scnn_u, scnn_r, scnn_l], dim=1)
        scnn_output = self.scnn_deconv(scnn_combined)
        res = self.res_branch(x)
        se = self.se_branch(x)
        return scnn_output * se + res

# LSHDNet implementation
class LSHDNet(nn.Module):
    def __init__(self):
        super(LSHDNet, self).__init__()
        backbone = resnet50(pretrained=True)
        self.stage1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.cm1 = ContextMixer(256)
        self.cm2 = ContextMixer(512)
        self.cm3 = ContextMixer(1024)
        self.cm4 = ContextMixer(2048)
        self.cm5 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feat0 = self.cm5(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        cm0 = feat0
        cm1 = self.cm1(feat1)
        cm2 = self.cm2(feat2)
        cm3 = self.cm3(feat3)
        cm4 = self.cm4(feat4)
        combined = torch.nn.functional.interpolate(cm4, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   torch.nn.functional.interpolate(cm3, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   torch.nn.functional.interpolate(cm2, size=cm0.shape[2:], mode='bilinear', align_corners=False) + \
                   torch.nn.functional.interpolate(cm1, size=cm0.shape[2:], mode='bilinear', align_corners=False) + cm0
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
        x = torch.nn.functional.relu(self.conv1(x))
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
        attention = torch.nn.functional.sigmoid(self.conv2(torch.nn.functional.relu(self.conv1(combined))))
        weighted_f1 = attention[:, :1, :, :] * f1
        weighted_f2 = attention[:, 1:, :, :] * f2
        return weighted_f1 + weighted_f2

# Main detection pipeline
class HighlightDetectionPipeline(nn.Module):
    def __init__(self):
        super(HighlightDetectionPipeline, self).__init__()
        self.lshdnet = LSHDNet()
        self.rm_modules = [RM() for _ in range(6)]
        self.afm = AFM()

    def forward(self, img):
        scales = [256, 512, 1024, 2048]
        resized_images = [Resize((s, s))(img) for s in scales]
        features = [self.lshdnet(resized) for resized in resized_images]
        f5 = self.rm_modules[0](features[3], features[2])
        f6 = self.rm_modules[1](f5, features[1])
        f7 = self.rm_modules[2](f6, features[0])
        f8 = self.rm_modules[3](features[1], features[0])
        f9 = self.rm_modules[4](f8, features[2])
        f10 = self.rm_modules[5](f9, features[3])
        return self.afm(f10, f7)

# Dataset Class
class HighlightDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = ToTensor()(self.images[idx])
        mask = ToTensor()(self.masks[idx])
        return image, mask

# Training Function
def train_pipeline(pipeline, train_loader, val_loader, epochs, lr, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(pipeline.parameters(), lr=lr)
    for epoch in range(epochs):
        pipeline.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = pipeline(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}")
        val_loss = validate_pipeline(pipeline, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

# Validation Function
def validate_pipeline(pipeline, val_loader, criterion, device):
    pipeline.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = pipeline(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Example Initialization and Training
train_images = [torch.rand(3, 2048, 2048) for _ in range(10)]
train_masks = [torch.randint(0, 2, (1, 2048, 2048)) for _ in range(10)]
val_images = [torch.rand(3, 2048, 2048) for _ in range(2)]
val_masks = [torch.randint(0, 2, (1, 2048, 2048)) for _ in range(2)]
train_dataset = HighlightDataset(train_images, train_masks)
val_dataset = HighlightDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = HighlightDetectionPipeline().to(device)
train_pipeline(pipeline, train_loader, val_loader, epochs=5, lr=1e-4, device=device)
