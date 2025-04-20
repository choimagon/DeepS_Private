import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 정의
class FaceInpaintingDataset(Dataset):
    def __init__(self, masked_paths, original_paths, transform=None):
        self.masked_paths = masked_paths
        self.original_paths = original_paths
        self.transform = transform

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        masked = Image.open(self.masked_paths[idx]).convert("RGB")
        original = Image.open(self.original_paths[idx]).convert("RGB")
        if self.transform:
            masked = self.transform(masked)
            original = self.transform(original)
        return masked, original

# U-Net Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(Generator, self).__init__()
        self.encoder1 = self._block(in_channels, features)
        self.encoder2 = self._block(features, features * 2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.encoder4 = self._block(features * 4, features * 8)

        self.bottleneck = self._block(features * 8, features * 16)

        self.decoder4 = self._block(features * 16 + features * 8, features * 8)
        self.decoder3 = self._block(features * 8 + features * 4, features * 4)
        self.decoder2 = self._block(features * 4 + features * 2, features * 2)
        self.decoder1 = self._block(features * 2 + features, features)

        self.final_layer = nn.Conv2d(features, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.decoder4(torch.cat([nn.functional.interpolate(bottleneck, size=enc4.shape[2:]), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([nn.functional.interpolate(dec4, size=enc3.shape[2:]), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([nn.functional.interpolate(dec3, size=enc2.shape[2:]), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([nn.functional.interpolate(dec2, size=enc1.shape[2:]), enc1], dim=1))

        return torch.tanh(self.final_layer(dec1))

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        channels = in_channels
        for feature in features:
            layers.append(nn.Conv2d(channels, feature, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = feature
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# (기존 내용 생략)

from torchvision.utils import make_grid

# 결과 저장 함수 수정 (train/val 모두 시각화)
def save_result(masked_tensor, pred_tensor, target_tensor, epoch, save_dir="results_unet"):
    os.makedirs(save_dir, exist_ok=True)
    masked_img = masked_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    pred_img = pred_tensor.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    target_img = target_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    masked_img = np.clip((masked_img + 1) / 2, 0, 1)
    pred_img = np.clip((pred_img + 1) / 2, 0, 1)
    target_img = np.clip((target_img + 1) / 2, 0, 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(masked_img)
    plt.title("Masked Input")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(pred_img)
    plt.title("Inpainted Output")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(target_img)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{epoch:03d}.png"))
    plt.close()

# 학습 코드 (train/val 분리 및 val 시각화 추가)
def train(masked_dir, original_dir, epochs=30, batch_size=4, lr=2e-4, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    original_paths = [os.path.join(original_dir, f) for f in all_files]

    train_m, val_m, train_o, val_o = train_test_split(masked_paths, original_paths, test_size=val_ratio, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FaceInpaintingDataset(train_m, train_o, transform)
    val_dataset = FaceInpaintingDataset(val_m, val_o, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        netG.train()
        for masked, original in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            masked, original = masked.to(device), original.to(device)

            with torch.no_grad():
                output_shape = netD(original).shape
                real_label = torch.ones(output_shape, device=device)
                fake_label = torch.zeros(output_shape, device=device)

            optimizerD.zero_grad()
            output_real = netD(original)
            lossD_real = criterion(output_real, real_label)
            fake = netG(masked)
            output_fake = netD(fake.detach())
            lossD_fake = criterion(output_fake, fake_label)
            lossD = (lossD_real + lossD_fake) * 0.5
            lossD.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            output_fake = netD(fake)
            lossG = criterion(output_fake, real_label) + 100 * criterionL1(fake, original)
            lossG.backward()
            optimizerG.step()

        # Validation 결과 시각화 (한 배치만 저장)
        netG.eval()
        with torch.no_grad():
            for val_masked, val_original in val_loader:
                val_masked = val_masked.to(device)
                val_original = val_original.to(device)
                val_output = netG(val_masked)
                save_result(val_masked, val_output, val_original, epoch+1)
                break  # 첫 배치만 저장

        torch.save(netG.state_dict(), "unet_generator.pth")

if __name__ == '__main__':
    train(masked_dir="00000_masked", original_dir="00000", epochs=200)
