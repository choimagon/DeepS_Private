import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ------------------------ Dataset ------------------------
class FaceInpaintingDataset(Dataset):
    def __init__(self, masked_paths, original_paths, mask_root, transform=None):
        self.masked_paths = masked_paths
        self.original_paths = original_paths
        self.mask_root = mask_root
        self.transform = transform

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        masked = Image.open(self.masked_paths[idx]).convert("RGB")
        original = Image.open(self.original_paths[idx]).convert("RGB")
        filename = os.path.basename(self.masked_paths[idx])
        mask = Image.open(os.path.join(self.mask_root, filename)).convert("L")

        if self.transform:
            masked = self.transform(masked)
            original = self.transform(original)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()

        return masked, original, mask

# ------------------------ Model ------------------------
class CoarseGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
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

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class RefineNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=5):
        super().__init__()
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(64))
        layers.append(nn.Conv2d(64, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))

class TwoStageInpaintingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = CoarseGenerator()
        self.refine = RefineNet()

    def forward(self, x):
        coarse_output = self.coarse(x)
        refined_output = self.refine(coarse_output)
        return coarse_output, refined_output

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

# ------------------------ Utility ------------------------
def prefill_mask_region(masked, mask, iterations=5):
    filled = masked.clone()
    kernel = torch.ones((masked.size(1), 1, 3, 3), device=masked.device) / 8.0
    for _ in range(iterations):
        neighbor_avg = F.conv2d(filled, kernel, padding=1, groups=masked.size(1))
        filled = filled * (1 - mask) + neighbor_avg * mask
    return filled

def save_result(masked_tensor, pred_tensor, target_tensor, epoch, save_dir="results_twostageNEW"):
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

# ------------------------ Training ------------------------
def train(masked_dir, original_dirs, mask_root, epochs=30, batch_size=8, lr=5e-4, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    original_paths = []
    for f in all_files:
        prefix = f.split('_')[0]
        for od in original_dirs:
            if prefix in od:
                original_paths.append(os.path.join(od, f[len(prefix)+1:]))
                break
        else:
            raise FileNotFoundError(f"{f} 에 해당하는 원본 이미지 경로를 찾을 수 없습니다.")

    train_m, val_m, train_o, val_o = train_test_split(masked_paths, original_paths, test_size=val_ratio, random_state=42)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = FaceInpaintingDataset(train_m, train_o, mask_root=mask_root, transform=transform)
    val_dataset = FaceInpaintingDataset(val_m, val_o, mask_root=mask_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    netG = TwoStageInpaintingModel().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    best_val_loss = float('inf')

    for epoch in range(epochs):
        netG.train()
        train_lossG, train_lossD = 0, 0

        for masked, original, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            masked, original, mask = masked.to(device), original.to(device), mask.to(device)
            prefilled = prefill_mask_region(masked, mask)

            # ------------------ Discriminator ------------------
            optimizerD.zero_grad()
            _, refined_output = netG(prefilled)
            output_real = netD(original)
            output_fake = netD(refined_output.detach())

            real_label = torch.ones_like(output_real, device=device)
            fake_label = torch.zeros_like(output_fake, device=device)

            lossD = 0.5 * (criterion(output_real, real_label) + criterion(output_fake, fake_label))
            lossD.backward()
            optimizerD.step()

            # ------------------ Generator ------------------
            optimizerG.zero_grad()
            coarse_output, refined_output = netG(prefilled)
            output_fake = netD(refined_output)

            # ⛔ 정규화 해제 없이 그대로 L1 계산 (-1~1 기준)
            lossG = criterion(output_fake, real_label) + \
                    50 * criterionL1(coarse_output, original) + \
                    100 * criterionL1(refined_output, original)
            lossG.backward()
            optimizerG.step()

            train_lossG += lossG.item()
            train_lossD += lossD.item()

        netG.eval()
        val_lossG = 0
        with torch.no_grad():
            val_lossG = 0
            val_count = 0
            for i, (val_masked, val_original, val_mask) in enumerate(val_loader):
                val_masked, val_original, val_mask = val_masked.to(device), val_original.to(device), val_mask.to(device)
                val_prefilled = prefill_mask_region(val_masked, val_mask)
                _, refined_output = netG(val_prefilled)

                # ✅ 정규화 상태 그대로 L1 Loss 계산 (-1 ~ 1 기준)
                val_loss = criterionL1(refined_output, val_original)
                val_lossG += val_loss.item()
                val_count += 1

                if i == 4:
                    save_result(val_masked, refined_output, val_original, epoch+1)

        print(f"[Epoch {epoch+1}] Train Loss G: {train_lossG/len(train_loader):.4f}, D: {train_lossD/len(train_loader):.4f} | Val Loss G: {val_lossG/val_count:.4f}")
        if (val_lossG / val_count) < best_val_loss:
            best_val_loss = val_lossG / val_count
            print(f"save - {best_val_loss}")
            torch.save(netG.state_dict(), "best_twostage_generatorT.pth")

if __name__ == '__main__':
    train(
        masked_dir="all_half_masked",
        original_dirs=[
            "data/image/00000",
            "data/image/01000",
            "data/image/02000",
            "data/image/03000"
        ],
        mask_root="all_half_mask_only",
        epochs=250
    )
