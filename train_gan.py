import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# --------- 커스텀 데이터셋 ---------
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

# --------- Generator (U-Net) ---------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.down1 = block(3, 64)
        self.down2 = block(64, 128)
        self.down3 = block(128, 256)
        self.down4 = block(256, 512)
        self.down5 = block(512, 512)

        self.up1 = up_block(512, 512)
        self.up2 = up_block(1024, 256)
        self.up3 = up_block(512, 128)
        self.up4 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        out = self.final(torch.cat([u4, d1], dim=1))
        return (out + 1) / 2

# --------- Discriminator ---------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(6, 64, norm=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        return self.model(input)

# --------- 이미지 저장 ---------
def save_result(masked_tensor, pred_tensor, epoch, save_dir="results_gan"):
    os.makedirs(save_dir, exist_ok=True)
    masked_img = masked_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    pred_img = pred_tensor.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    masked_img = np.clip(masked_img, 0, 1)
    pred_img = np.clip(pred_img, 0, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(masked_img)
    plt.title("Masked Input")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img)
    plt.title("Predicted Output")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{epoch:03d}.png"))
    plt.close()

# --------- 학습 함수 ---------
def train_model(masked_dir, original_dir, epochs=30, batch_size=4, lr=2e-4, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    original_paths = [os.path.join(original_dir, f) for f in all_files]

    train_m, val_m, train_o, val_o = train_test_split(
        masked_paths, original_paths, test_size=val_ratio, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = FaceInpaintingDataset(train_m, train_o, transform)
    val_dataset = FaceInpaintingDataset(val_m, val_o, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for masked, real in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            masked, real = masked.to(device), real.to(device)

            pred_shape = discriminator(masked, real).shape
            valid = torch.ones(pred_shape, device=device)
            fake = torch.zeros_like(valid)

            optimizer_g.zero_grad()
            gen_img = generator(masked)
            pred_fake = discriminator(masked, gen_img)
            loss_g_gan = criterion_gan(pred_fake, valid)
            loss_g_l1 = criterion_l1(gen_img, real)
            loss_g = loss_g_gan + 100 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            pred_real = discriminator(masked, real)
            loss_d_real = criterion_gan(pred_real, valid)
            pred_fake = discriminator(masked, gen_img.detach())
            loss_d_fake = criterion_gan(pred_fake, fake)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()

        generator.eval()
        with torch.no_grad():
            for i, (masked, _) in enumerate(val_loader):
                masked = masked.to(device)
                output = generator(masked)
                save_result(masked, output, epoch+1)
                break

        print(f"[Epoch {epoch+1}] Generator Loss: {loss_g.item():.4f}, Discriminator Loss: {loss_d.item():.4f}")

if __name__ == "__main__":
    train_model(masked_dir="00000mask", original_dir="00000", epochs=30)