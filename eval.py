import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_score
from skimage.metrics import structural_similarity as ssim_score
import lpips
from pytorch_fid import fid_score
from sklearn.model_selection import train_test_split

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

        return masked, original, mask, filename

# ------------------------ CBAM ------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

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
        dec4 = self.decoder4(torch.cat([F.interpolate(bottleneck, size=enc4.shape[2:]), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:]), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:]), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:]), enc1], dim=1))
        return torch.tanh(self.final_layer(dec1))

class ResidualBlockCBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.cbam = CBAM(channels)

    def forward(self, x):
        return x + self.cbam(self.block(x))

class RefineNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(ResidualBlockCBAM(64))
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

# ------------------------ Utility ------------------------
def prefill_mask_region(masked, mask, iterations=5):
    filled = masked.clone()
    kernel = torch.ones((masked.size(1), 1, 3, 3), device=masked.device) / 8.0
    for _ in range(iterations):
        neighbor_avg = F.conv2d(filled, kernel, padding=1, groups=masked.size(1))
        filled = filled * (1 - mask) + neighbor_avg * mask
    return filled

def save_test_result(masked_tensor, pred_tensor, target_tensor, filename, save_dir="results_eval_cbam"):
    os.makedirs(save_dir, exist_ok=True)
    masked_img = masked_tensor.cpu().permute(1, 2, 0).numpy()
    pred_img = pred_tensor.cpu().permute(1, 2, 0).detach().numpy()
    target_img = target_tensor.cpu().permute(1, 2, 0).numpy()
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
    plt.title("Output")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(target_img)
    plt.title("GT")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# ------------------------ Evaluation ------------------------
def evaluate(model_path, masked_dir, original_dirs, mask_root, sample_size=100, val_ratio=0.2):
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
            raise FileNotFoundError(f"{f} 가 최종 원본이 없습니다.")

    _, val_masked_paths, _, val_original_paths = train_test_split(masked_paths, original_paths, test_size=val_ratio, random_state=42)

    if len(val_masked_paths) < sample_size:
        raise ValueError("벨리데이션 데이터가 사용 생일 발치 받을 수 없습니다.")

    indices = random.sample(range(len(val_masked_paths)), sample_size)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = FaceInpaintingDataset(val_masked_paths, val_original_paths, mask_root, transform)
    subset = Subset(val_dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    netG = TwoStageInpaintingModel().to(device)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    pred_dir, gt_dir = "eval_cbam_pred", "eval_cbam_gt"
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    with torch.no_grad():
        for masked, original, mask, filename in tqdm(loader):
            masked, original, mask = masked.to(device), original.to(device), mask.to(device)
            prefilled = prefill_mask_region(masked, mask)
            _, output = netG(prefilled)

            save_test_result(masked[0], output[0], original[0], str(filename[0]))
            out_np = ((output[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            gt_np = ((original[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

            total_psnr += psnr_score(gt_np, out_np, data_range=255)
            total_ssim += ssim_score(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)
            total_lpips += loss_fn_lpips(output, original).item()

            filename_str = filename[0] if isinstance(filename[0], str) else filename[0].decode("utf-8") if hasattr(filename[0], 'decode') else str(filename[0])

            Image.fromarray(out_np).save(os.path.join(pred_dir, filename_str))
            Image.fromarray(gt_np).save(os.path.join(gt_dir, filename_str))

    print(f"✅ PSNR: {total_psnr / sample_size:.4f}")
    print(f"✅ SSIM: {total_ssim / sample_size:.4f}")
    print(f"✅ LPIPS: {total_lpips / sample_size:.4f}")
    fid = fid_score.calculate_fid_given_paths([gt_dir, pred_dir], batch_size=8, device=device, dims=2048)
    print(f"✅ FID: {fid:.4f}")

if __name__ == '__main__':
    evaluate(
        model_path="best_twostage_generator_cbam90.pth",
        masked_dir="all_half_masked",
        original_dirs=[
            "data/image/00000",
            "data/image/01000",
            "data/image/02000",
            "data/image/03000"
        ],
        mask_root="all_half_mask_only",
        sample_size=100,
        val_ratio=0.2
    )