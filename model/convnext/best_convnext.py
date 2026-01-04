import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from PIL import ImageFilter
from io import BytesIO
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler


# ================== Seed ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== JPEG Artifact Aug ==================
class RandomJPEGCompression:
    def __init__(self, p=0.25, quality_min=25, quality_max=75):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        q = random.randint(self.quality_min, self.quality_max)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ================== Recommended Degradation Augs ==================
class RandomDownUp:
    """
    Downsample -> Upsample to simulate low-resolution + upsampling artifacts.
    """
    def __init__(self, p=0.35, min_scale=0.22, max_scale=0.55, down_interp=InterpolationMode.BILINEAR):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.down_interp = down_interp

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        nw, nh = max(8, int(w * s)), max(8, int(h * s))

        img_small = transforms.functional.resize(img, (nh, nw), interpolation=self.down_interp)
        up_interp = random.choice([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
        img_back = transforms.functional.resize(img_small, (h, w), interpolation=up_interp)
        return img_back


class RandomMotionBlur:
    """
    Simple blur to mimic motion blur (stable PIL-based).
    """
    def __init__(self, p=0.25, radius_min=1.0, radius_max=3.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        r = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomGamma:
    """
    Random gamma adjustment to simulate exposure/overexposure and highlight clipping.
    gamma < 1: brighter, gamma > 1: darker
    """
    def __init__(self, p=0.35, gamma_min=0.65, gamma_max=1.55):
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr, 0, 1) ** gamma
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomGaussianNoise:
    """
    Add Gaussian noise in PIL space before ToTensor.
    """
    def __init__(self, p=0.20, sigma_min=2.0, sigma_max=12.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# ================== Transforms ==================
def build_train_transform(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.70, 1.0),
            ratio=(0.90, 1.12),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),

        # ✅ Low-res simulation
        RandomDownUp(p=0.35, min_scale=0.22, max_scale=0.55),

        # ✅ Color/exposure
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.03),
        RandomGamma(p=0.35, gamma_min=0.65, gamma_max=1.55),

        # ✅ Blur / pseudo motion blur
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
        RandomMotionBlur(p=0.25, radius_min=1.0, radius_max=3.0),

        # ✅ JPEG artifacts (more aggressive than before)
        RandomJPEGCompression(p=0.30, quality_min=15, quality_max=70),

        # ✅ Noise
        RandomGaussianNoise(p=0.20, sigma_min=2.0, sigma_max=12.0),

        transforms.ToTensor(),

        # ✅ Occlusion
        transforms.RandomErasing(p=0.30, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ================== Focal Loss ==================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ================== Head subtype (파일명 토큰 기반, 정확) ==================
def infer_head_subtype_from_filename(path: str) -> str:
    path = path.lower()
    if "raincoat" in path:
        return "raincoat"
    if "bald" in path:
        return "bald"
    if "cap" in path:
        return "cap"
    return "other"


# ================== Weighted sampling (핵심) ==================
def make_sample_weights(dataset: ImageFolder, class_to_idx: dict):
    idx_head = class_to_idx.get("head", None)
    idx_helmet = class_to_idx.get("helmet", None)
    if idx_head is None or idx_helmet is None:
        raise ValueError(f"class_to_idx에 'head'/'helmet'이 없습니다: {class_to_idx}")

    labels = [y for _, y in dataset.samples]
    counts = np.bincount(labels, minlength=len(class_to_idx)).astype(np.float32)
    counts[counts == 0] = 1.0

    # 기본: 클래스 균형
    base_class_w = 1.0 / counts

    # ✅ head 서브타입별 샘플링 배수
    MULT_RAINCOAT = 4.0
    MULT_BALD     = 3.0
    MULT_CAP      = 1.5
    MULT_OTHER    = 1.0

    sample_weights = []
    for path, y in dataset.samples:
        w = float(base_class_w[y])

        if y == idx_head:
            st = infer_head_subtype_from_filename(path)
            if st == "raincoat":
                w *= MULT_RAINCOAT
            elif st == "bald":
                w *= MULT_BALD
            elif st == "cap":
                w *= MULT_CAP
            else:
                w *= MULT_OTHER

        sample_weights.append(w)

    return sample_weights


def print_train_subtype_stats(dataset: ImageFolder, indices):
    idx_head = dataset.class_to_idx.get("head", None)
    if idx_head is None:
        print("⚠️ 'head' 클래스가 없어 서브타입 통계 생략")
        return

    c = {"raincoat": 0, "bald": 0, "cap": 0, "other": 0, "helmet": 0}
    for i in indices:
        path, y = dataset.samples[i]
        if y == idx_head:
            c[infer_head_subtype_from_filename(path)] += 1
        else:
            c["helmet"] += 1

    total = len(indices)
    print("📌 Train subset 구성(파일명 토큰 기반):")
    for k, v in c.items():
        print(f"   - {k:>8}: {v:6d} ({(v/max(1,total))*100:5.1f}%)")


# ================== Train / Val ==================
def run_epoch(model, loader, criterion, optimizer, device, train=True, use_amp=True):
    if train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and train))

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item())
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=float(loss.item()))

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


# ================== Main ==================
if __name__ == "__main__":
    # ======= 설정(추천 기본값) =======
    SEED = 42
    IMG_SIZE = 224

    BATCH_SIZE = 32
    EPOCHS = 40
    LR = 2e-4
    WEIGHT_DECAY = 1e-4

    LABEL_SMOOTHING = 0.03
    FOCAL_GAMMA = 1.5

    EARLY_STOPPING_PATIENCE = 20
    NUM_WORKERS = 2
    PIN_MEMORY = True

    DATA_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\12월23일\new_crops_300"
    SAVE_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\save_pt_convnext_v3_224"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ==============================

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"✅ Device: {device} | AMP: {use_amp}")

    # dataset split용
    full_for_split = ImageFolder(DATA_DIR)
    print(f"📌 class_to_idx: {full_for_split.class_to_idx}")

    # split
    train_size = int(0.8 * len(full_for_split))
    val_size = len(full_for_split) - train_size
    g = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(full_for_split, [train_size, val_size], generator=g)

    # 실제 학습용 dataset(각자 transform 다르게)
    train_ds = ImageFolder(DATA_DIR, transform=build_train_transform(IMG_SIZE))
    val_ds = ImageFolder(DATA_DIR, transform=build_val_transform(IMG_SIZE))

    train_subset = torch.utils.data.Subset(train_ds, train_idx.indices)
    val_subset = torch.utils.data.Subset(val_ds, val_idx.indices)

    # (선택) 학습 subset의 서브타입 분포 출력
    print_train_subtype_stats(train_ds, train_idx.indices)

    # ----- Weighted sampler (핵심) -----
    base_weights = make_sample_weights(train_ds, train_ds.class_to_idx)
    subset_weights = [base_weights[i] for i in train_idx.indices]
    sampler = WeightedRandomSampler(subset_weights, num_samples=len(subset_weights), replacement=True)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ======= Model =======
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    model = model.to(device)

    # ======= Loss =======
    labels_train = [train_ds.samples[i][1] for i in train_idx.indices]
    counts = np.bincount(labels_train, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_w = (counts.sum() / (2.0 * counts))
    class_w = torch.tensor(class_w, dtype=torch.float32, device=device)
    print(f"⚖️ class weights: {class_w.detach().cpu().numpy()}")

    criterion = FocalLoss(alpha=class_w, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    # ======= Optimizer =======
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ======= Train Loop =======
    best_val_loss = float("inf")
    best_path = os.path.join(SAVE_DIR, "best.pt")
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n🟢 Epoch {epoch}/{EPOCHS} | lr={optimizer.param_groups[0]['lr']:.2e}")

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, use_amp=use_amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, use_amp=use_amp)
        scheduler.step()

        print(f"  ✅ Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}%")
        print(f"  📊 Val  : loss={va_loss:.4f} acc={va_acc*100:.2f}%")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"🏆 Best 업데이트! val_loss={best_val_loss:.4f} 저장 → {best_path}")
            patience = 0
        else:
            patience += 1
            print(f"⏳ EarlyStop counter {patience}/{EARLY_STOPPING_PATIENCE}")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping. best val_loss={best_val_loss:.4f}")
            break

    print(f"\n✅ 완료. Best 모델: {best_path}")
