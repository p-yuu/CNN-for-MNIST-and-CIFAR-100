from pathlib import Path

import torch
import torch.nn as nn

from data_loader import get_cifar100_loaders
from model import CIFAR100CNNV3


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_cifar100_v4_ema.pth"

BATCH_SIZE = 128


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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, _ = get_cifar100_loaders(batch_size=BATCH_SIZE,augment=False)
    model = CIFAR100CNNV3().to(device)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print("\nEvaluation Results")
    print("-" * 45)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc * 100:.2f}%")
    print(
        f"Accuracy Gap: {(train_acc - val_acc) * 100:.2f} pp"
    )


if __name__ == "__main__":
    main()
