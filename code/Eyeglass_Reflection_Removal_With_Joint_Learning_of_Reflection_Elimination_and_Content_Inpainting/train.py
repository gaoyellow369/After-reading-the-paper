import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util import AverageMeters, write_loss, progress_bar
from specdata import TrainDataset, TestDataset
from losses import VGGLoss, GradientLoss, MaskLoss
import os

# MTNetWithRFM Model Definition
class Encoder(nn.Module):
    def __init__(self, input_channels, features=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class MemoryModule(nn.Module):
    def __init__(self, feature_dim, memory_size):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))

    def forward(self, features):
        B, C, H, W = features.size()
        features_flattened = features.view(B, C, -1).permute(0, 2, 1)
        attn_weights = F.softmax(torch.matmul(features_flattened, self.memory.T), dim=-1)
        memory_output = torch.matmul(attn_weights, self.memory)
        return memory_output.permute(0, 2, 1).view(B, C, H, W)

class RFM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(RFM, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.partial_conv = nn.Conv2d(input_channels * 2, output_channels, kernel_size=3, padding=1)

    def forward(self, Ie, Ii, D):
        M = self.weight_generator(D)
        Ie_weighted = Ie * M
        Ii_weighted = Ii * (1 - M)
        Icat = torch.cat((Ie_weighted, Ii_weighted), dim=1)
        M_avg = F.avg_pool2d(M, kernel_size=3, stride=1, padding=1)
        I_out = self.partial_conv(Icat) / (M_avg + 1e-8)
        I_out = torch.where(M_avg > 0, I_out, torch.zeros_like(I_out))
        return I_out

class MTNetWithRFM(nn.Module):
    def __init__(self):
        super(MTNetWithRFM, self).__init__()
        self.encoder = Encoder(input_channels=3)
        self.detector = Decoder(input_channels=64, output_channels=1)
        self.memory_module = MemoryModule(feature_dim=64, memory_size=512)
        self.inpainting_decoder = Decoder(input_channels=128, output_channels=3)
        self.elimination_decoder = Decoder(input_channels=64, output_channels=3)
        self.residual_block = ResidualBlock(channels=64)
        self.rfm = RFM(input_channels=64, output_channels=3)

    def forward(self, x):
        F = self.encoder(x)
        D = self.detector(F)
        Ie = self.elimination_decoder(self.residual_block(F))
        memory_features = self.memory_module(F)
        Ii = self.inpainting_decoder(torch.cat((F, memory_features), dim=1))
        I_out = self.rfm(Ie, Ii, D)
        return I_out, D, Ie, Ii

# Loss function setup
class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        self.reconstruction_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.gradient_loss = GradientLoss()
        self.mask_loss = MaskLoss()

    def forward(self, outputs, targets, masks, delta_map):
        I_out, D, Ie, Ii = outputs
        recon_loss = self.reconstruction_loss(I_out, targets)
        vgg_loss = self.vgg_loss(I_out, targets)
        gradient_loss = self.gradient_loss(I_out, targets)
        mask_loss = self.mask_loss(D, delta_map)

        total_loss = recon_loss + 0.1 * vgg_loss + 0.1 * gradient_loss + 0.1 * mask_loss
        return total_loss, {
            'reconstruction_loss': recon_loss.item(),
            'vgg_loss': vgg_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'mask_loss': mask_loss.item()
        }

# Training setup
def train_model(model, dataloader, optimizer, loss_wrapper, device):
    model.train()
    avg_meters = AverageMeters()
    for i, data in enumerate(dataloader):
        inputs, targets, masks, delta_map = data['input'].to(device), data['target'].to(device), data['mask'].to(device), data['map'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, loss_dict = loss_wrapper(outputs, targets, masks, delta_map)
        loss.backward()
        optimizer.step()

        avg_meters.update(loss_dict)
        progress_bar(i, len(dataloader), str(avg_meters))

# Evaluation setup
def evaluate_model(model, dataloader, loss_wrapper, device):
    model.eval()
    avg_meters = AverageMeters()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, targets, masks, delta_map = data['input'].to(device), data['target'].to(device), data['mask'].to(device), data['map'].to(device)
            outputs = model(inputs)
            _, loss_dict = loss_wrapper(outputs, targets, masks, delta_map)
            avg_meters.update(loss_dict)
    return avg_meters

# Main training script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = TrainDataset(opt=None, datadir='./dataset/train', path1='specular', path2='gt')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    val_dataset = TestDataset(datadir='./dataset/val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model initialization
    model = MTNetWithRFM().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # Loss wrapper
    loss_wrapper = LossWrapper().to(device)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_model(model, train_loader, optimizer, loss_wrapper, device)
        val_metrics = evaluate_model(model, val_loader, loss_wrapper, device)

        print(f"Validation Loss: {val_metrics}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoints/MTNet_epoch_{epoch + 1}.pth")
