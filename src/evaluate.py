from pathlib import Path

import torch
import torch.nn as nn

from data_loader import get_mnist_loaders
from model import MNISTCNN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_mnist_cnn.pth"


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

    return total_loss / total, correct / total, correct, total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, _, test_loader = get_mnist_loaders(batch_size=64)

    model = MNISTCNN().to(device)
    state_dict = torch.load(MODEL_PATH,map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, correct, total = evaluate(model, test_loader, criterion, device)

    print("\nMNIST CNN Test Results")
    print("-----------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Correct Predictions: {correct}/{total}")


if __name__ == "__main__":
    main()
