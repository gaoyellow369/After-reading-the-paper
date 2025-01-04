import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder Block
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


# Decoder Block
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


# Residual Block
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


# Memory Module
class MemoryModule(nn.Module):
    def __init__(self, feature_dim, memory_size):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
    
    def forward(self, features):
        # Flatten spatial dimensions for memory computation
        B, C, H, W = features.size()
        features_flattened = features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        # Attention mechanism
        attn_weights = F.softmax(torch.matmul(features_flattened, self.memory.T), dim=-1)
        memory_output = torch.matmul(attn_weights, self.memory)  # (B, H*W, C)
        
        # Reshape back to original spatial dimensions
        return memory_output.permute(0, 2, 1).view(B, C, H, W)


# Result Fusion Module (RFM)
class RFM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(RFM, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.partial_conv = nn.Conv2d(input_channels * 2, output_channels, kernel_size=3, padding=1)
        self.bias = nn.Parameter(torch.zeros(output_channels))

    def forward(self, Ie, Ii, D):
        # Generate weight map M
        M = self.weight_generator(D)
        
        # Compute weighted results
        Ie_weighted = Ie * M
        Ii_weighted = Ii * (D - M)
        Icat = torch.cat((Ie_weighted, Ii_weighted), dim=1)  # Concatenate along channel dimension
        
        # Compute local weight average
        M_avg = F.avg_pool2d(M, kernel_size=3, stride=1, padding=1)
        
        # Partial convolution for final output
        I_out = self.partial_conv(Icat) / (M_avg + 1e-8)
        I_out = torch.where(M_avg > 0, I_out, torch.zeros_like(I_out))
        
        return I_out


# MTNet with RFM
class MTNetWithRFM(nn.Module):
    def __init__(self):
        super(MTNetWithRFM, self).__init__()
        self.encoder = Encoder(input_channels=3)  # Encoder shared across branches
        self.detector = Decoder(input_channels=64, output_channels=1)  # Detection branch
        self.memory_module = MemoryModule(feature_dim=64, memory_size=512)  # Memory module for inpainting branch
        self.inpainting_decoder = Decoder(input_channels=128, output_channels=3)  # Inpainting branch decoder
        self.elimination_decoder = Decoder(input_channels=64, output_channels=3)  # Elimination branch decoder
        self.residual_block = ResidualBlock(channels=64)  # Residual block for elimination branch
        self.rfm = RFM(input_channels=64, output_channels=3)  # Result Fusion Module
        
    def forward(self, x):
        # Shared Encoder
        F = self.encoder(x)
        
        # Detection Branch
        D = self.detector(F)
        
        # Elimination Branch
        Ie = self.elimination_decoder(self.residual_block(F))
        
        # Inpainting Branch
        memory_features = self.memory_module(F)
        Ii = self.inpainting_decoder(torch.cat((F, memory_features), dim=1))
        
        # Result Fusion
        I_out = self.rfm(Ie, Ii, D)
        
        return I_out, D, Ie, Ii


# Example Usage
if __name__ == "__main__":
    # Create model
    model = MTNetWithRFM()
    
    # Generate dummy input
    input_image = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels (RGB), 256x256 resolution
    
    # Forward pass
    I_out, D, Ie, Ii = model(input_image)
    
    # Print shapes of outputs
    print(f"Final Output Shape (I_out): {I_out.shape}")
    print(f"Detection Result Shape (D): {D.shape}")
    print(f"Elimination Result Shape (Ie): {Ie.shape}")
    print(f"Inpainting Result Shape (Ii): {Ii.shape}")
