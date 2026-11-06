from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn

from data_loader import get_mnist_loaders
from model import MNISTCNN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
RESULT_DIR = BASE_DIR / "outputs" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42


def set_seed(seed):
    random.seed(seed)                       # Python seed
    np.random.seed(seed)                    # Numpy seed
    torch.manual_seed(seed)                 # CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    # GPU seed


def train_one_epoch(model, loader, criterion, optimizer, device):   # criterion: Loss Function
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               # 清除梯度
        outputs = model(images)             # 向前傳播
        loss = criterion(outputs, labels)   # 計算 loss
        loss.backward()                     # 反向傳播
        optimizer.step()                    # 更新權重

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)             # 預測 -> 找分數最高的類別
        correct += (predictions == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()    # Python Decorator -> 在這個函數執行期間，不需要追蹤梯度
def evaluate(model, loader, criterion, device):         # 驗證模型
    model.eval()    # 評估模型

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


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    model_path = MODEL_DIR / "best_mnist_cnn.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

    csv_path = RESULT_DIR / "mnist_training_history.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining completed!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved Model: {model_path}")
    print(f"Saved History: {csv_path}")


if __name__ == "__main__":
    main()
