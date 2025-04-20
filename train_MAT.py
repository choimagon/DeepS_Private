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

# --------- Advanced MAT 구조 (MAE 스타일 + full output) ---------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MATInpainting(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, max_len=self.num_patches)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8), num_layers=6)

        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, x):
        B = x.shape[0]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)

        N = patches.shape[1]
        num_mask = int(N * self.mask_ratio)
        rand_indices = torch.rand(B, N, device=x.device).argsort(dim=1)
        keep_indices = rand_indices[:, :-num_mask]
        mask_indices = rand_indices[:, -num_mask:]

        visible = torch.gather(patches, 1, keep_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        mask_tokens = self.mask_token.expand(B, num_mask, self.embed_dim)
        full_seq = torch.cat([visible, mask_tokens], dim=1)

        sorted_indices = torch.cat([keep_indices, mask_indices], dim=1)
        restore_indices = sorted_indices.argsort(dim=1)
        full_seq = torch.gather(full_seq, 1, restore_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        full_seq = self.pos_embed(full_seq)
        encoded = self.transformer(full_seq)
        decoded = self.decoder(encoded).view(B, self.num_patches, 3, self.patch_size, self.patch_size)

        # 재조합 (모든 패치 출력 사용)
        recon_img = torch.zeros(B, 3, self.img_size, self.img_size, device=x.device)
        idx = 0
        for i in range(0, self.img_size, self.patch_size):
            for j in range(0, self.img_size, self.patch_size):
                recon_img[:, :, i:i+self.patch_size, j:j+self.patch_size] = decoded[:, idx]
                idx += 1

        return recon_img

# --------- 이미지 저장 ---------
def save_result(masked_tensor, pred_tensor, epoch, save_dir="results"):
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
    print(f"Device: {device}")

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

    model = MATInpainting().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked, original in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            masked, original = masked.to(device), original.to(device)
            optimizer.zero_grad()
            output = model(masked).clamp(0, 1)
            loss = criterion(output, original)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (masked, original) in enumerate(val_loader):
                masked, original = masked.to(device), original.to(device)
                output = model(masked).clamp(0, 1)
                loss = criterion(output, original)
                val_loss += loss.item()
                if i == 0:
                    save_result(masked, output, epoch+1)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_mat_transformer.pth')
            print("Best 모델 저장 완료")

if __name__ == "__main__":
    train_model(masked_dir="00000mask", original_dir="00000", epochs=30)
