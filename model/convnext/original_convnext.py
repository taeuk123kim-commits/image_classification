import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ✅ ConvNeXt-Tiny import
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


# ================== 하이퍼파라미터 ==================
BATCH_SIZE = 4
EPOCHS = 80
NUM_CLASSES = 2
LEARNING_RATE = 1e-4

# DATA_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\new_crops_300"
DATA_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\12월23일\new_crops_300"
SAVE_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\save_pt_convnext"
SAVE_INTERVAL = 10

SEED = 42
USE_CLASS_WEIGHT = True
NUM_WORKERS = 2
PIN_MEMORY = True
# ===================================================


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def count_labels_from_subset(subset, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)
    for idx in subset.indices:
        _, y, _ = subset.dataset[idx]
        counts[y] += 1
    return counts


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    full_dataset_for_split = ImageFolderWithPaths(root=DATA_DIR, transform=None)

    class_idx_to_name = {v: k for k, v in full_dataset_for_split.class_to_idx.items()}
    print("\n📌 클래스 매핑:")
    for k in sorted(class_idx_to_name.keys()):
        print(f"  {k}: {class_idx_to_name[k]}")

    train_size = int(0.8 * len(full_dataset_for_split))
    val_size = len(full_dataset_for_split) - train_size

    g = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_dataset_for_split, [train_size, val_size], generator=g)

    print(f"\n✅ Train: {train_size}개 | Val: {val_size}개")

    train_tf = build_train_transform()
    val_tf = build_val_transform()

    train_base = ImageFolderWithPaths(root=DATA_DIR, transform=train_tf)
    val_base = ImageFolderWithPaths(root=DATA_DIR, transform=val_tf)

    train_dataset = torch.utils.data.Subset(train_base, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_base, val_subset.indices)

    train_counts = count_labels_from_subset(train_dataset, NUM_CLASSES)
    val_counts = count_labels_from_subset(val_dataset, NUM_CLASSES)

    print("\n📊 클래스 분포(Train):")
    for i in range(NUM_CLASSES):
        print(f"  {i}({class_idx_to_name[i]}): {train_counts[i]}")

    print("\n📊 클래스 분포(Val):")
    for i in range(NUM_CLASSES):
        print(f"  {i}({class_idx_to_name[i]}): {val_counts[i]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ================== ✅ 모델: ConvNeXt-Tiny ==================
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # ConvNeXt classifier 구조: Sequential( LayerNorm, Flatten, Linear(768->1000) )
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(device)
    # ============================================================

    if USE_CLASS_WEIGHT:
        total = train_counts.sum()
        weights = total / (NUM_CLASSES * np.maximum(train_counts, 1))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"\n⚖️ Class Weights 적용: {weights.detach().cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ✅ best.pt 저장용 변수
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n🟢 [Epoch {epoch}/{EPOCHS}]")

        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc="  🔁 Training", leave=False)
        for images, labels, paths in train_loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_loop.set_postfix(loss=float(loss.item()))

        avg_loss = running_loss / max(len(train_loader), 1)
        acc = correct / max(total, 1)
        print(f"✅ Epoch {epoch}: Train Loss={avg_loss:.4f} | Train Acc={acc*100:.2f}%")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_loop = tqdm(val_loader, desc="  🧪 Validation", leave=False)
        with torch.no_grad():
            for images, labels, paths in val_loop:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_loop.set_postfix(loss=float(loss.item()))

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"📊 Validation: Val Loss={avg_val_loss:.4f} | Val Acc={val_acc*100:.2f}%")

        # ✅ Best Save (val_acc 기준)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🏆 Best 갱신! val_acc={best_val_acc*100:.2f}% → 저장: {best_path}")

        # ---- Save interval ----
        if epoch % SAVE_INTERVAL == 0:
            save_fname = f"convnext_tiny__epoch_{epoch}.pth"
            save_path = os.path.join(SAVE_DIR, save_fname)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "train_acc": acc,
                "val_acc": val_acc,
                "class_to_idx": train_base.class_to_idx,
                "seed": SEED,
            }, save_path)
            print(f"\n✅ 모델 저장 완료: {save_path}")

    final_name = f"convnext_tiny__epoch_{EPOCHS}.pth"
    final_path = os.path.join(SAVE_DIR, final_name)
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": train_base.class_to_idx,
        "seed": SEED,
    }, final_path)
    print(f"\n✅ 최종 모델 저장 완료: {final_path}")
    print(f"🏁 Best 모델: {os.path.join(SAVE_DIR, 'best.pt')} (val_acc 최고: {best_val_acc*100:.2f}%)")

