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

# --------- 모델 정의 ---------
class InpaintingDeepLab(nn.Module):
    def __init__(self):
        super().__init__()
        self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.deeplab.classifier = nn.Sequential(
            nn.Conv2d(2048, 224, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(224, 3, kernel_size=1)
        )

    def forward(self, x):
        return self.deeplab(x)["out"]

# --------- Perceptual Loss ---------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# --------- SSIM Loss ---------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return 1 - ssim.mean()

# --------- 이미지 저장 ---------
def save_result(masked_tensor, pred_tensor, epoch, save_dir="results_deeplab"):
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
def train_model(masked_dir, original_dir, epochs=30, batch_size=4, lr=1e-4, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Device: {device}")

    all_files = sorted([f for f in os.listdir(masked_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    masked_paths = [os.path.join(masked_dir, f) for f in all_files]
    original_paths = [os.path.join(original_dir, f) for f in all_files]

    train_m, val_m, train_o, val_o = train_test_split(
        masked_paths, original_paths, test_size=val_ratio, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = FaceInpaintingDataset(train_m, train_o, transform)
    val_dataset = FaceInpaintingDataset(val_m, val_o, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = InpaintingDeepLab().to(device)
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    ssim_loss_fn = SSIMLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked, original in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            masked, original = masked.to(device), original.to(device)
            optimizer.zero_grad()
            output = model(masked).clamp(0, 1)

            loss_l1 = l1_loss_fn(output, original)
            loss_perc = perceptual_loss_fn(output, original)
            loss_ssim = ssim_loss_fn(output, original)
            loss = 0.6 * loss_l1 + 0.2 * loss_perc + 0.2 * loss_ssim

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (masked, original) in enumerate(val_loader):
                masked, original = masked.to(device), original.to(device)
                output = model(masked).clamp(0, 1)
                loss_l1 = l1_loss_fn(output, original)
                loss_perc = perceptual_loss_fn(output, original)
                loss_ssim = ssim_loss_fn(output, original)
                loss = 0.6 * loss_l1 + 0.2 * loss_perc + 0.2 * loss_ssim
                val_loss += loss.item()
                if i == 0:
                    save_result(masked, output, epoch+1)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_deeplab_inpaint.pth')
            print("💾 Best 모델 저장 완료")

if __name__ == "__main__":
    train_model(masked_dir="00000mask", original_dir="00000", epochs=30)
