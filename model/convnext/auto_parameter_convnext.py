import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from io import BytesIO
from tqdm import tqdm

import optuna
from optuna.trial import TrialState

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from torch.utils.data import DataLoader, random_split, Dataset
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


# ================== Transforms ==================
def build_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_head_strong_transform(
    img_size=224,
    jitter_b=0.35, jitter_c=0.35, jitter_s=0.18, jitter_h=0.03,
    jpeg_p=0.20, jpeg_qmin=30, jpeg_qmax=70,
    erase_p=0.45,
):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h),
        RandomJPEGCompression(p=jpeg_p, quality_min=jpeg_qmin, quality_max=jpeg_qmax),
        transforms.ToTensor(),
        transforms.RandomErasing(p=erase_p, scale=(0.02, 0.20), ratio=(0.25, 4.0), value="random"),
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
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.0):
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


# ================== Head subtype ==================
def infer_head_subtype(path: str) -> str:
    p = path.lower()
    if "raincoat" in p:
        return "raincoat"
    if "bald" in p:
        return "bald"
    if "cap" in p:
        return "cap"
    return "other"


# ================== Train dataset wrapper ==================
class SubtypeAugDataset(Dataset):
    def __init__(self, root, normal_tf, strong_tf, strong_subtypes=("cap", "raincoat", "bald")):
        self.base = ImageFolder(root)
        self.normal_tf = normal_tf
        self.strong_tf = strong_tf
        self.strong_subtypes = set(strong_subtypes)

        if "head" not in self.base.class_to_idx or "helmet" not in self.base.class_to_idx:
            raise ValueError(f"클래스 폴더에 head/helmet 이 있어야 합니다. class_to_idx={self.base.class_to_idx}")

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        path, y = self.base.samples[idx]
        img = default_loader(path)

        cls_name = self.base.classes[y]
        if cls_name == "head":
            st = infer_head_subtype(path)
            if st in self.strong_subtypes:
                img = self.strong_tf(img)
            else:
                img = self.normal_tf(img)
        else:
            img = self.normal_tf(img)

        return img, y


# ================== Weighted sampling ==================
def make_sample_weights(dataset: ImageFolder, class_to_idx: dict,
                        mult_raincoat=4.0, mult_bald=3.0, mult_cap=1.5, mult_other=1.0):
    idx_head = class_to_idx.get("head", None)
    idx_helmet = class_to_idx.get("helmet", None)
    if idx_head is None or idx_helmet is None:
        raise ValueError(f"class_to_idx에 'head'/'helmet'이 없습니다: {class_to_idx}")

    labels = [y for _, y in dataset.samples]
    counts = np.bincount(labels, minlength=len(class_to_idx)).astype(np.float32)
    counts[counts == 0] = 1.0

    base_class_w = 1.0 / counts

    sample_weights = []
    for path, y in dataset.samples:
        w = float(base_class_w[y])

        if y == idx_head:
            st = infer_head_subtype(path)
            if st == "raincoat":
                w *= mult_raincoat
            elif st == "bald":
                w *= mult_bald
            elif st == "cap":
                w *= mult_cap
            else:
                w *= mult_other

        sample_weights.append(w)

    return sample_weights


# ================== Train / Val epoch ==================
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


@torch.no_grad()
def eval_metrics_binary(model, loader, device, positive_class=1):
    model.eval()
    tp = tn = fp = fn = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(1)

        y = labels
        p = preds

        if positive_class == 1:
            tp += ((p == 1) & (y == 1)).sum().item()
            tn += ((p == 0) & (y == 0)).sum().item()
            fp += ((p == 1) & (y == 0)).sum().item()
            fn += ((p == 0) & (y == 1)).sum().item()
        else:
            tp += ((p == 0) & (y == 0)).sum().item()
            tn += ((p == 1) & (y == 1)).sum().item()
            fp += ((p == 0) & (y == 1)).sum().item()
            fn += ((p == 1) & (y == 0)).sum().item()

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ================== Optuna param space ==================
def trial_param_space(trial: optuna.Trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 2.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.08),

        "mult_raincoat": trial.suggest_float("mult_raincoat", 1.0, 6.0),
        "mult_bald": trial.suggest_float("mult_bald", 1.0, 6.0),
        "mult_cap": trial.suggest_float("mult_cap", 1.0, 6.0),
        "mult_other": trial.suggest_float("mult_other", 1.0, 2.5),

        "jitter_b": trial.suggest_float("jitter_b", 0.15, 0.45),
        "jitter_c": trial.suggest_float("jitter_c", 0.15, 0.45),
        "jitter_s": trial.suggest_float("jitter_s", 0.05, 0.25),
        "jitter_h": trial.suggest_float("jitter_h", 0.00, 0.05),
        "jpeg_p": trial.suggest_float("jpeg_p", 0.0, 0.35),
        "erase_p": trial.suggest_float("erase_p", 0.0, 0.55),
    }
f

# ================== Main ==================
if __name__ == "__main__":
    # ======= 설정 =======
    SEED = 42
    IMG_SIZE = 224

    BATCH_SIZE = 32

    # ✅ Optuna: trial 빠르게 평가할 epoch / trial 개수
    QUICK_EPOCHS = 12
    N_TRIALS = 25
    QUICK_EARLY_STOP_PATIENCE = 3

    # ✅ BEST PARAMS 찾은 뒤, 최종 FULL 학습 epoch
    EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 20

    POSITIVE_CLASS = 1  # 1=helmet, 0=head
    BEST_BY = "acc"      # "f1" or "acc" or "loss"

    NUM_WORKERS = 0
    PIN_MEMORY = True

    DATA_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\12월23일\new_crops_300"
    SAVE_DIR = r"D:\20125_1216_태욱_만듦(끝나고삭제)\save_pt_convnext_optuna"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 쓰기 테스트(권한/백신 문제 조기 확인)
    test_path = os.path.join(SAVE_DIR, "_write_test.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("ok")
    print("✅ SAVE_DIR write ok:", SAVE_DIR)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"✅ Device: {device} | AMP: {use_amp}")

    # split indices 고정
    full_for_split = ImageFolder(DATA_DIR)
    print(f"📌 class_to_idx: {full_for_split.class_to_idx}")
    assert set(full_for_split.class_to_idx.keys()) >= {"head", "helmet"}, "head/helmet 폴더가 필요합니다."

    train_size = int(0.8 * len(full_for_split))
    val_size = len(full_for_split) - train_size
    g = torch.Generator().manual_seed(SEED)
    train_subset_for_split, val_subset_for_split = random_split(full_for_split, [train_size, val_size], generator=g)

    train_indices = train_subset_for_split.indices
    val_indices = val_subset_for_split.indices

    # ============ Trial objective ============
    def objective(trial: optuna.Trial):
        set_seed(SEED)

        params = trial_param_space(trial)

        # datasets/transforms
        train_ds = SubtypeAugDataset(
            DATA_DIR,
            normal_tf=build_train_transform(IMG_SIZE),
            strong_tf=build_head_strong_transform(
                IMG_SIZE,
                jitter_b=params["jitter_b"],
                jitter_c=params["jitter_c"],
                jitter_s=params["jitter_s"],
                jitter_h=params["jitter_h"],
                jpeg_p=params["jpeg_p"],
                erase_p=params["erase_p"],
            ),
            strong_subtypes=("cap", "raincoat", "bald")
        )
        val_ds = ImageFolder(DATA_DIR, transform=build_val_transform(IMG_SIZE))

        train_subset = torch.utils.data.Subset(train_ds, train_indices)
        val_subset = torch.utils.data.Subset(val_ds, val_indices)

        # sampler
        base_weights = make_sample_weights(
            train_ds.base, train_ds.base.class_to_idx,
            mult_raincoat=params["mult_raincoat"],
            mult_bald=params["mult_bald"],
            mult_cap=params["mult_cap"],
            mult_other=params["mult_other"]
        )
        subset_weights = [base_weights[i] for i in train_indices]
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

        # model
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 2)
        model = model.to(device)

        # loss/opt/sched
        criterion = FocalLoss(gamma=params["focal_gamma"], label_smoothing=params["label_smoothing"])
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QUICK_EPOCHS, eta_min=1e-6)

        best_score = -1e9
        patience = 0

        for epoch in range(1, QUICK_EPOCHS + 1):
            run_epoch(model, train_loader, criterion, optimizer, device, train=True, use_amp=use_amp)
            va_loss, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False, use_amp=use_amp)

            metrics = eval_metrics_binary(model, val_loader, device, positive_class=POSITIVE_CLASS)
            scheduler.step()

            if BEST_BY == "loss":
                score = -va_loss
            elif BEST_BY == "acc":
                score = metrics["acc"]
            else:
                score = metrics["f1"]

            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if score > best_score + 1e-6:
                best_score = score
                patience = 0
            else:
                patience += 1
                if patience >= QUICK_EARLY_STOP_PATIENCE:
                    break

        return float(best_score)

    # ============ Run Optuna ============
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    print(f"\n🚀 Optuna 시작: trials={N_TRIALS} | best_by={BEST_BY}")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n====================")
    print("🏁 Optuna 완료")
    print(f"Best value: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("====================\n")

    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Trials: complete={len(complete_trials)}, pruned={len(pruned_trials)}, total={len(study.trials)}")

    # ============ FULL TRAIN with best params ============
    best_params = study.best_params

    # datasets/transforms (best params)
    train_ds = SubtypeAugDataset(
        DATA_DIR,
        normal_tf=build_train_transform(IMG_SIZE),
        strong_tf=build_head_strong_transform(
            IMG_SIZE,
            jitter_b=best_params["jitter_b"],
            jitter_c=best_params["jitter_c"],
            jitter_s=best_params["jitter_s"],
            jitter_h=best_params["jitter_h"],
            jpeg_p=best_params["jpeg_p"],
            erase_p=best_params["erase_p"],
        ),
        strong_subtypes=("cap", "raincoat", "bald")
    )
    val_ds = ImageFolder(DATA_DIR, transform=build_val_transform(IMG_SIZE))

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    base_weights = make_sample_weights(
        train_ds.base, train_ds.base.class_to_idx,
        mult_raincoat=best_params["mult_raincoat"],
        mult_bald=best_params["mult_bald"],
        mult_cap=best_params["mult_cap"],
        mult_other=best_params["mult_other"]
    )
    subset_weights = [base_weights[i] for i in train_indices]
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

    # model
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    model = model.to(device)

    # loss/opt/sched
    criterion = FocalLoss(gamma=best_params["focal_gamma"], label_smoothing=best_params["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ======= Train Loop (epoch별 저장 + best 저장) =======
    best_path = os.path.join(SAVE_DIR, "best.pt")
    epoch_dir = os.path.join(SAVE_DIR, "epochs")
    os.makedirs(epoch_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_acc = -1.0
    patience = 0

    print("\n🔥 Best params로 FULL 학습 시작")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🟢 Epoch {epoch}/{EPOCHS} | lr={optimizer.param_groups[0]['lr']:.2e}")

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, use_amp=use_amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, use_amp=use_amp)

        metrics = eval_metrics_binary(model, val_loader, device, positive_class=POSITIVE_CLASS)

        scheduler.step()

        print(f"  ✅ Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}%")
        print(f"  📊 Val  : loss={va_loss:.4f} acc={va_acc*100:.2f}%")
        print(f"  📌 Val metrics(positive={POSITIVE_CLASS}): "
              f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} | "
              f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

        # ✅ (1) epoch별 무조건 저장
        epoch_path = os.path.join(epoch_dir, f"epoch_{epoch:03d}.pt")
        try:
            torch.save(model.state_dict(), epoch_path)
            print(f"💾 Epoch 저장 → {epoch_path}")
        except Exception as e:
            print("❌ epoch 저장 실패:", repr(e))
            raise

        # ----- best 저장 기준 선택 -----
        if BEST_BY == "loss":
            improved = va_loss < best_val_loss
        elif BEST_BY == "acc":
            improved = metrics["acc"] > best_val_acc
        else:  # "f1"
            improved = metrics["f1"] > best_val_f1

        # ✅ (2) best 저장(개선 시)
        if improved:
            try:
                torch.save(model.state_dict(), best_path)
                print(f"🏆 Best 업데이트! (by {BEST_BY}) 저장 → {best_path}")
            except Exception as e:
                print("❌ best 저장 실패:", repr(e))
                raise

            patience = 0
            best_val_loss = min(best_val_loss, va_loss)
            best_val_acc = max(best_val_acc, metrics["acc"])
            best_val_f1 = max(best_val_f1, metrics["f1"])
        else:
            patience += 1
            print(f"⏳ EarlyStop counter {patience}/{EARLY_STOPPING_PATIENCE}")

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping. (best_by={BEST_BY})")
            break

    print(f"\n✅ 완료.")
    print(f"   - Best 모델: {best_path}")
    print(f"   - Epoch별 모델 폴더: {epoch_dir}")
