import argparse
import csv
import math
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from data_loader import get_data_loaders
from model import CIFAR100CNNV3


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "outputs" / "models"
RESULT_DIR = BASE_DIR / "outputs" / "results"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
CHECKPOINT_PATH = MODEL_DIR / "last_checkpoint.pth"
HISTORY_PATH = RESULT_DIR / "training_history.csv"


SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MIN_LR = 1e-4
WARMUP_EPOCHS = 5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(images, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = (lam * images + (1.0 - lam) * images[index])

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def cutmix_data(images, labels, alpha=CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    labels_a = labels
    labels_b = labels[index]

    _, _, height, width = images.shape

    cut_ratio = np.sqrt(1.0 - lam)

    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = np.random.randint(width)
    center_y = np.random.randint(height)

    x1 = np.clip(center_x - cut_width // 2, 0, width)
    x2 = np.clip(center_x + cut_width // 2, 0, width)

    y1 = np.clip(center_y - cut_height // 2, 0, height)
    y2 = np.clip(center_y + cut_height // 2, 0, height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = (images[index, :, y1:y2, x1:x2])

    box_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - box_area / (width * height)

    return mixed_images, labels_a, labels_b, lam


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress_bar = tqdm(loader,desc="Training",leave=False,dynamic_ncols=True)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if random.random() < CUTMIX_PROB:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
        else:
            images, labels_a, labels_b, lam = mixup_data(images, labels)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = (lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b))
        loss.backward()
        optimizer.step()
        predictions = outputs.argmax(dim=1)

        correct_a = (predictions == labels_a).sum().item()
        correct_b = (predictions == labels_b).sum().item()

        batch_correct = (lam * correct_a + (1.0 - lam) * correct_b)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples * 100:.2f}%"
        )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100.0

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(loader,desc="Validation",leave=False,dynamic_ncols=True)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples * 100:.2f}%"
        )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100.0

    return avg_loss, avg_acc


def create_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS

        progress = ((epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = MIN_LR / LEARNING_RATE

        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def append_history(row):
    file_exists = HISTORY_PATH.exists()

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "mix_train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "epoch_time_sec",
    ]

    with open(HISTORY_PATH,"a",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true",help="Resume from last_checkpoint.pth")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = get_data_loaders(batch_size=BATCH_SIZE,seed=SEED)

    print(f"Number of classes: {num_classes}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    model = CIFAR100CNNV3(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY,nesterov=True)

    scheduler = create_scheduler(optimizer)

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH,map_location=device,weights_only=False)

        if checkpoint["num_classes"] != num_classes:
            raise ValueError("Number of classes changed. Cannot resume this checkpoint.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]

        print(f"Resuming from epoch {start_epoch}")

    else:
        # Start a new training history
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()

    print("\nTraining started!")
    print("-" * 65)

    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model,train_loader,optimizer,criterion,device)
        val_loss, val_acc = evaluate(model,val_loader,criterion,device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(),BEST_MODEL_PATH)

        scheduler.step()
        epoch_time = time.time() - start_time

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
            },
            CHECKPOINT_PATH
        )

        append_history(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "mix_train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Mix Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Best: {best_val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
