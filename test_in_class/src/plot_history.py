from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_PATH = BASE_DIR / "outputs" / "results" / "training_history.csv"
FIGURE_DIR = BASE_DIR / "outputs" / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_history():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    history = pd.read_csv(HISTORY_PATH)

    required_columns = [
        "epoch",
        "train_loss",
        "mix_train_acc",
        "val_loss",
        "val_acc",
        "learning_rate",
    ]

    missing = [
        col for col in required_columns
        if col not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}"
        )

    return history



def plot_loss(history):
    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
        linewidth=2
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = FIGURE_DIR / "loss_curve.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved: {save_path}")


def plot_accuracy(history):
    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["mix_train_acc"],
        label="Mix Train Accuracy",
        linewidth=2
    )

    plt.plot(
        history["epoch"],
        history["val_acc"],
        label="Validation Accuracy",
        linewidth=2
    )

    # Mark best validation accuracy
    best_idx = history["val_acc"].idxmax()

    best_epoch = int(history.loc[best_idx, "epoch"])
    best_acc = history.loc[best_idx, "val_acc"]

    plt.scatter(
        best_epoch,
        best_acc,
        color="red",
        s=65,
        zorder=5
    )

    plt.annotate(
        f"Best: {best_acc:.2f}% (Epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(10, -25),
        textcoords="offset points"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = FIGURE_DIR / "accuracy_curve.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved: {save_path}")



def main():
    history = load_history()

    print(f"Loaded {len(history)} epochs")

    plot_loss(history)
    plot_accuracy(history)

    plt.show()


if __name__ == "__main__":
    main()
