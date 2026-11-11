import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import CIFAR100CNNV3
from data_loader import IMAGE_SIZE, MEAN, STD


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.pth"
MAPPING_PATH = BASE_DIR / "outputs" / "results" / "idx_to_class.json"
OUTPUT_PATH = BASE_DIR / "outputs" / "results" / "411385019.txt"

BATCH_SIZE = 128
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class TestImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = Path(folder)
        self.transform = transform

        if not self.folder.exists():
            raise FileNotFoundError(f"Test folder not found: {self.folder}")

        self.image_paths = sorted(
            path for path in self.folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise ValueError(f"No test images found in {self.folder}")

        stems = [path.stem for path in self.image_paths]

        if len(stems) != len(set(stems)):
            raise ValueError("Duplicate image names detected after removing extensions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return image_tensor, image_path.stem


@torch.no_grad()
def predict(model, loader, idx_to_class, device):
    model.eval()

    results = []

    progress_bar = tqdm(loader,desc="Predicting",dynamic_ncols=True)

    for images, image_names in progress_bar:
        images = images.to(device, non_blocking=True)

        logits_original = model(images)

        flipped_images = torch.flip(images, dims=[3])
        logits_flipped = model(flipped_images)

        probs_original = F.softmax(logits_original, dim=1)
        probs_flipped = F.softmax(logits_flipped, dim=1)

        # probs_tta = (0.5 * probs_original + 0.5 * probs_flipped)

        # predictions = probs_tta.argmax(dim=1).cpu().tolist()
        predictions = probs_original.argmax(dim=1).cpu().tolist()

        for image_name, pred in zip(image_names, predictions):
            class_name = idx_to_class[str(pred)]

            results.append(f"{image_name} {class_name}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {MAPPING_PATH}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)

    num_classes = len(idx_to_class)

    print(f"Number of classes: {num_classes}")

    model = CIFAR100CNNV3(num_classes=num_classes).to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH,map_location=device,weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dataset = TestImageDataset(folder=TEST_DIR,transform=test_transform)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())

    print(f"Test images: {len(test_dataset)}")

    results = predict(model,test_loader,idx_to_class,device)

    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    with open(OUTPUT_PATH,"w",encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")

    print("\nPrediction completed!")
    print(f"Predicted images: {len(results)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
