import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# UNet 구조 (깊이 증가)
# -------------------------------
class DeepUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.block(3, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.enc4 = self.block(256, 512)

        # Bottleneck
        self.bottleneck = self.block(512, 1024)

        # Decoder
        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)

        # Final
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d3 = self.up3(d4 + e4)
        d2 = self.up2(d3 + e3)
        d1 = self.up1(d2 + e2)

        out = self.final(d1 + e1)
        return out

# -------------------------------
# Dataset 클래스
# -------------------------------
class FaceInpaintingDataset(Dataset):
    def __init__(self, masked_paths, original_paths, transform=None):
        self.masked_paths = masked_paths
        self.original_paths = original_paths
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        masked = Image.open(self.masked_paths[idx]).convert("RGB")
        original = Image.open(self.original_paths[idx]).convert("RGB")
        if self.transform:
            masked = self.transform(masked)
            original = self.transform(original)
        mask = (masked == 0).float().mean(dim=0, keepdim=True)
        return masked, original, mask

# -------------------------------
# Forward noise
# -------------------------------
def q_sample(x0, t, noise):
    return (1 - t) * x0 + t * noise

# -------------------------------
# RePaint step
# -------------------------------
def repaint_step(model, x, mask, t, device):
    denoised = model(x)
    x = mask * denoised + (1 - mask) * x
    noise = torch.randn_like(x)
    return (1 - t) * x + t * noise

# -------------------------------
# 시각화
# -------------------------------
def save_visualization(masked, predicted, original, epoch, save_dir="repaint_unet_results"): 
    os.makedirs(save_dir, exist_ok=True)
    masked_img = masked.permute(1, 2, 0).cpu().numpy()
    pred_img = predicted.permute(1, 2, 0).detach().cpu().numpy()
    original_img = original.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(9, 3))
    for i, (img, title) in enumerate(zip([masked_img, pred_img, original_img], ["Masked", "Predicted", "GT"])):
        plt.subplot(1, 3, i+1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close()

# -------------------------------
# Training + Inference
# -------------------------------
def train_repaint_unet(masked_dir, original_dir, epochs=30, batch_size=4, jump_len=10, resample_steps=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Device: {device}")

    all_files = sorted([f for f in os.listdir(masked_dir) if f.endswith(('.jpg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    original_paths = [os.path.join(original_dir, f) for f in all_files]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_len = int(len(masked_paths) * 0.2)
    train_set = FaceInpaintingDataset(masked_paths[val_len:], original_paths[val_len:], transform)
    val_set = FaceInpaintingDataset(masked_paths[:val_len], original_paths[:val_len], transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = DeepUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for masked, original, mask in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            masked, original, mask = masked.to(device), original.to(device), mask.to(device)
            noise = torch.randn_like(original)
            t = torch.rand(masked.size(0), 1, 1, 1, device=device)
            x_noised = q_sample(original, t, noise)
            pred = model(x_noised)
            loss = F.mse_loss(pred * mask, original * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for masked, original, mask in val_loader:
                masked, original, mask = masked.to(device), original.to(device), mask.to(device)
                x = torch.randn_like(original)
                for t in reversed(range(1, 50)):
                    t_scaled = torch.tensor(t / 50, device=device)
                    x = repaint_step(model, x, mask, t_scaled, device)

                    # resampling jump
                    for _ in range(resample_steps):
                        for j in range(jump_len):
                            forward_t = min(49, t + j)
                            t_fwd = torch.tensor(forward_t / 50, device=device)
                            x = q_sample(x, t_fwd, torch.randn_like(x))
                        for j in range(jump_len):
                            back_t = max(1, t - j)
                            t_bwd = torch.tensor(back_t / 50, device=device)
                            x = repaint_step(model, x, mask, t_bwd, device)
                save_visualization(masked[0], x[0], original[0], epoch + 1)
                break
    torch.save(model.state_dict(), "repaint_deep_unet.pth")
    print("✅ 저장 완료: repaint_deep_unet.pth")

# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    train_repaint_unet("00000mask", "00000", epochs=30)
