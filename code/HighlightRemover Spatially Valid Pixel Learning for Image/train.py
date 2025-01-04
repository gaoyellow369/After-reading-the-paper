import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import HighlightRemovalNet  # 模型定义
from reflect_dataset import CEILDataset  # 数据集加载器
from transforms import to_tensor  # 数据预处理
import argparse

# 自定义损失函数
class HighlightLoss(nn.Module):
    def __init__(self):
        super(HighlightLoss, self).__init__()
        self.color_loss = nn.MSELoss()  # 颜色一致性损失
        self.texture_loss = nn.L1Loss()  # 纹理一致性损失

    def forward(self, pred, target):
        # 计算颜色损失（MSE）
        color_loss = self.color_loss(pred, target)
        # 计算纹理损失（基于梯度差异）
        pred_dx, pred_dy = torch.gradient(pred)
        target_dx, target_dy = torch.gradient(target)
        texture_loss = self.texture_loss(pred_dx, target_dx) + self.texture_loss(pred_dy, target_dy)
        return color_loss + 0.1 * texture_loss  # 总损失加权求和

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='高光去除网络的训练脚本')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=8, help='训练的批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练的总轮数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存模型的目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备（cuda 或 cpu）')
    return parser.parse_args()

# 训练函数
def train():
    args = parse_args()

    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 数据集与数据加载器
    train_dataset = CEILDataset(datadir=args.data_dir, enable_transforms=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 模型、优化器与损失函数
    model = HighlightRemovalNet(in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HighlightLoss()

    # 开始训练
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)  # 带高光的输入图像
            targets = batch['target_t'].to(device)  # 目标真实图像

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'第 {epoch + 1}/{args.epochs} 轮, 第 {i + 1}/{len(train_loader)} 批, 损失: {loss.item():.4f}')

        print(f'第 {epoch + 1}/{args.epochs} 轮完成. 平均损失: {epoch_loss / len(train_loader):.4f}')

        # 保存模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch + 1}.pth'))

if __name__ == '__main__':
    train()
