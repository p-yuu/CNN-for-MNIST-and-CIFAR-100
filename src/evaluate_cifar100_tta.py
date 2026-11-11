from pathlib import Path

import torch
import torch.nn.functional as F
import argparse

from data_loader import get_cifar100_loaders
from model import CIFAR100CNNV3


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_cifar100_v8.pth"

BATCH_SIZE = 128


@torch.no_grad()
def evaluate_tta(model, loader, device):
    model.eval()

    weights = [0.3, 0.4, 0.5, 0.6, 0.7]

    correct = {w: 0 for w in weights}
    correct_original = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits_original = model(images)
        logits_flipped = model(torch.flip(images, dims=[3]))

        probs_original = F.softmax(logits_original, dim=1)
        probs_flipped = F.softmax(logits_flipped, dim=1)

        pred_original = probs_original.argmax(dim=1)
        correct_original += (pred_original == labels).sum().item()

        for w in weights:
            probs_tta = (
                w * probs_original
                + (1 - w) * probs_flipped
            )

            predictions = probs_tta.argmax(dim=1)
            correct[w] += (predictions == labels).sum().item()

        total += labels.size(0)

    print(f"Original Accuracy: {correct_original / total * 100:.2f}%")

    for w in weights:
        acc = correct[w] / total * 100
        print(f"Original Weight {w:.1f}: {acc:.2f}%")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["val", "test"],
        default="val",
        help="Dataset to evaluate: val or test"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = CIFAR100CNNV3().to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    _, val_loader, test_loader = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        augment=True,
        augmentation="v5"
    )

    if args.dataset == "val":
        loader = val_loader
        print("Dataset: Validation")
    else:
        loader = test_loader
        print("Dataset: Test")

    print("\nWeighted TTA Evaluation Results")
    print("---------------------------------------------")

    evaluate_tta(model, loader, device)



if __name__ == "__main__":
    main()
