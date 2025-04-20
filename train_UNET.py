import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

# --------- Attention Block ---------
class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --------- 커스텀 데이터셋 ---------
class FaceInpaintingDataset(Dataset):
    def __init__(self, masked_paths, mask_only_paths, original_paths, transform=None):
        self.masked_paths = masked_paths
        self.mask_only_paths = mask_only_paths
        self.original_paths = original_paths
        self.transform = transform

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        masked = Image.open(self.masked_paths[idx]).convert("RGB")
        mask_only = Image.open(self.mask_only_paths[idx]).convert("L")  # grayscale
        original = Image.open(self.original_paths[idx]).convert("RGB")

        if self.transform:
            masked = self.transform(masked)
            mask_only = self.transform(mask_only)
            original = self.transform(original)

        return masked, mask_only, original

# --------- Attention U-Net 모델 ---------
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(4, 64)   # masked(3) + mask(1)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionBlock(512, 512, 256)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionBlock(256, 256, 128)
        self.dec3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(128, 128, 64)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, masked, mask):
        x = torch.cat([masked, mask], dim=1)  # (B, 4, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        a4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.upconv3(d4)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.upconv2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.upconv1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return torch.sigmoid(self.final(d1))

# --------- SSIM Loss ---------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return 1 - ssim.mean()

# --------- VGG Perceptual Loss ---------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# --------- 이미지 저장 ---------
def save_result(masked_tensor, pred_tensor, epoch, save_dir="results_AttnUNet"):
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
def train_model(masked_dir, mask_only_dir, original_dir, epochs=30, batch_size=32, lr=1e-4, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    mask_only_paths = [os.path.join(mask_only_dir, f) for f in all_files]
    original_paths = [os.path.join(original_dir, f) for f in all_files]

    train_m, val_m, train_mask, val_mask, train_o, val_o = train_test_split(
        masked_paths, mask_only_paths, original_paths, test_size=val_ratio, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = FaceInpaintingDataset(train_m, train_mask, train_o, transform)
    val_dataset = FaceInpaintingDataset(val_m, val_mask, val_o, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AttentionUNet().to(device)
    l1_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked, mask_only, original in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            masked, mask_only, original = masked.to(device), mask_only.to(device), original.to(device)
            optimizer.zero_grad()
            output = model(masked, mask_only).clamp(0, 1)
            loss = 0.6 * l1_loss(output, original) + 0.2 * perceptual_loss(output, original) + 0.2 * ssim_loss(output, original)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (masked, mask_only, original) in enumerate(val_loader):
                masked, mask_only, original = masked.to(device), mask_only.to(device), original.to(device)
                output = model(masked, mask_only).clamp(0, 1)
                loss = 0.6 * l1_loss(output, original) + 0.2 * perceptual_loss(output, original) + 0.2 * ssim_loss(output, original)
                val_loss += loss.item()
                if i == 0:
                    save_result(masked, output, epoch+1)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_attention_unet.pth')
            print("✅ Best 모델 저장 완료")

# --------- 실행 ---------
if __name__ == "__main__":
    train_model(
        masked_dir="00000_masked",
        mask_only_dir="00000_mask_only",
        original_dir="00000",
        epochs=100
    )
