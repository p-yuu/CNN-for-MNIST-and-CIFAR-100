from pathlib import Path
from tqdm.auto import tqdm
import csv
import random
import time
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from data_loader import get_cifar100_loaders
from model import CIFAR100CNNV6


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
RESULT_DIR = BASE_DIR / "outputs" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SEED = 42
WARMUP_EPOCHS = 5
MIN_LR = 1e-5

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS

    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))

    return (MIN_LR / LEARNING_RATE + (1 - MIN_LR / LEARNING_RATE) * cosine)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    indices = torch.randperm(batch_size,device=images.device)

    mixed_images = (lam * images + (1 - lam) * images[indices])

    labels_a = labels
    labels_b = labels[indices]

    return mixed_images, labels_a, labels_b, lam


def cutmix_data(images, labels, alpha=1.0):     # alpha：控制裁切區域大小
    lam = np.random.beta(alpha, alpha)          # 隨機產生 lambda (ex: 0.7 表原始圖片保留 70% 的面積，另一張圖片提供 30%)

    batch_size, _, height, width = images.shape
    indices = torch.randperm(batch_size, device=images.device)  # 隨機打亂圖片順序

    cut_ratio = np.sqrt(1.0 - lam)      # 計算裁切區域大小
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)       # 隨機決定裁切中心點
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = (images[indices, :, y1:y2, x1:x2])   # 疊圖

    patch_area = (x2 - x1) * (y2 - y1)          # 重新計算 lambda
    lam = 1.0 - patch_area / (width * height)

    return mixed_images, labels, labels[indices], lam


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, ema):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader,desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",unit="batch",leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
        outputs = model(images)
        loss = (lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b))

        loss.backward()
        optimizer.step()
        ema.update(model)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (lam * (predictions == labels_a).sum().item() + (1 - lam) * (predictions == labels_b).sum().item())
        total += batch_size

        progress_bar.set_postfix(loss=f"{total_loss / total:.4f}",acc=f"{correct / total * 100:.2f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        self.decay = decay
        self.num_updates = 0

        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1

        # EMA warmup
        decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        for ema_param, model_param in zip(self.ema.parameters(),model.parameters()):
            ema_param.lerp_(model_param.detach(),1.0 - decay)

        # Copy BatchNorm buffers
        for ema_buffer, model_buffer in zip(self.ema.buffers(),model.buffers()):
            ema_buffer.copy_(model_buffer)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = get_cifar100_loaders(batch_size=BATCH_SIZE,augment=True,augmentation="v5")
    model = CIFAR100CNNV6().to(device)
    ema = ModelEMA(model, decay=0.999)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)         # 在 EPOCHS 的週期長度中，逐漸將 LR 降到 1e-5

    best_val_acc = -1.0
    best_ema_val_acc = -1.0

    best_epoch = 0
    best_ema_epoch = 0

    history = []

    model_path = MODEL_DIR / "best_cifar100_v6.pth"
    csv_path = RESULT_DIR / "cifar100_v6_history.csv"

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model,train_loader,train_criterion,optimizer,device,epoch,ema)
        val_loss, val_acc = evaluate(model,val_loader,eval_criterion,device)
        ema_val_loss, ema_val_acc = evaluate(ema.ema,val_loader,eval_criterion,device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - start_time
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if ema_val_acc > best_ema_val_acc:
            best_ema_val_acc = ema_val_acc
            best_ema_epoch = epoch
            torch.save(ema.ema.state_dict(),MODEL_DIR / "best_cifar100_v6_ema.pth")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "ema_val_loss": ema_val_loss,
            "ema_val_acc": ema_val_acc,
            "epoch_time_sec": epoch_time,
            "learning_rate": current_lr
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"EMA Val Loss: {ema_val_loss:.4f} | "
            f"EMA Val Acc: {ema_val_acc * 100:.2f}% | "
            f"Best Val Acc: {best_val_acc * 100:.2f}% | "
            f"Best EMA Acc: {best_ema_val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s",
            flush=True
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print("\nTraining completed!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best EMA Epoch: {best_ema_epoch}")
    print(f"Best EMA Validation Accuracy: {best_ema_val_acc * 100:.2f}%")
    print(f"Saved Model: {model_path}")
    print(f"Saved EMA Model: {MODEL_DIR / 'best_cifar100_v6_ema.pth'}")
    print(f"Saved History: {csv_path}")


if __name__ == "__main__":
    main()
