import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
RESULT_DIR = BASE_DIR / "outputs" / "results"

IMAGE_SIZE = 32

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def create_class_mapping():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

    class_names = sorted([
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ])

    # if len(class_names) != 20:
    #     raise ValueError(f"Expected 20 classes, found {len(class_names)}")

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    idx_to_class = {
        str(idx): class_name
        for class_name, idx in class_to_idx.items()
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / "class_to_idx.json","w",encoding="utf-8") as f:
        json.dump(class_to_idx,f,indent=4,ensure_ascii=False)

    with open(RESULT_DIR / "idx_to_class.json","w",encoding="utf-8") as f:
        json.dump(idx_to_class,f,indent=4,ensure_ascii=False)

    print(f"Number of classes: {len(class_names)}")
    print("Class mapping saved.")

    return class_to_idx, idx_to_class



def get_data_loaders(batch_size=128, val_ratio=0.1, seed=42):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_dataset = datasets.ImageFolder(root=TRAIN_DIR)
    num_classes = len(full_dataset.classes)

    if num_classes < 2:
        raise ValueError(f"At least 2 classes are required, found {num_classes}")

    class_to_idx, _ = create_class_mapping()

    if full_dataset.class_to_idx != class_to_idx:
        raise ValueError("Class mapping mismatch!")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset),generator=generator).tolist()

    val_size = int(len(indices) * val_ratio)
    if val_size == 0:
        raise ValueError("Validation set is empty!")

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=train_transform)
    val_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    train_loader, val_loader, num_classes = get_data_loaders()

    print(f"Number of classes: {num_classes}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    images, labels = next(iter(train_loader))

    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Label range: {labels.min().item()} ~ {labels.max().item()}")
