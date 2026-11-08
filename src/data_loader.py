import pickle
from pathlib import Path
from PIL import Image
 
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split 
from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

def get_mnist_loaders(batch_size = 64):
    transform = transforms.ToTensor()       # 將 MNIST 圖片轉成 PyTorch Tensor

    full_train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)

    train_dataset, val_dataset = random_split(full_train_dataset, [55000,5000], generator=torch.Generator().manual_seed(42))
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LocalCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.transform = transform

        filename = "train" if train else "test"

        with open(self.root / filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")

        self.images = data[b"data"].reshape(-1, 3, 32, 32)      # Image data: (N, 3072) -> (N, 3, 32, 32)
        self.labels = data[b"fine_labels"]                      # Use 100 fine-grained classes

    def __len__(self):              # 照片總數
        return len(self.labels)

    def __getitem__(self, index):   # 根據索引取得一張圖片及其標籤
        image = self.images[index].transpose(1, 2, 0)           # (3, 32, 32) -> (32, 32, 3)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        return image, label


def get_cifar100_loaders(batch_size=128, augment=False):
    cifar_dir = DATA_DIR / "cifar-100-python"

    if not cifar_dir.is_dir():
        raise FileNotFoundError(f"CIFAR-100 directory not found: {cifar_dir}")

    for filename in ("train", "test"):
        if not (cifar_dir / filename).is_file():
            raise FileNotFoundError(f"Missing CIFAR-100 file: {cifar_dir / filename}")

    mean = (0.5071, 0.4867, 0.4408)         # RGB 平均值
    std = (0.2675, 0.2565, 0.2761)          # RGB 標準差

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),       # 隨機裁切
            transforms.RandomHorizontalFlip(p=0.5),     # 隨機水平翻轉
            transforms.ToTensor(),                      # 轉換成 Tensor
            transforms.Normalize(mean, std),
            transforms.RandomErasing(                   # 隨機遮蔽
                p=0.25,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value=0
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Validation / Test: no augmentation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = LocalCIFAR100(root=cifar_dir,train=True,transform=train_transform)
    full_val_dataset = LocalCIFAR100(root=cifar_dir,train=True,transform=eval_transform)
    test_dataset = LocalCIFAR100(root=cifar_dir,train=False,transform=eval_transform)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_train_dataset),generator=generator).tolist()      # 打亂索引

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size=128, augment=True)

    images, labels = next(iter(train_loader))

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Testing samples:", len(test_loader.dataset))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Label range:", labels.min().item(), labels.max().item())