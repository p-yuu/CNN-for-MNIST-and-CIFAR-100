import torch
import torch.nn as nn

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
                                                                                    # in_channels=1   ：1 個 Channel (灰階)
        self.features = nn.Sequential(                                              # out_channels=32 ：32 個可學習的 Filters
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),    # kernel_size=3   ：3 × 3 kernel
            nn.ReLU(),                                                              # padding=1       ：zero padding
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),         # Layer1：128 個神經元
            nn.ReLU(),
            nn.Linear(128,10)                   # Layer2：10 個神經元
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x 

class CIFAR100CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CIFAR100CNNV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),                             # normalize
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),                                # 2 * 2 取 max
            nn.Dropout2d(0.1),                              # 隨機將 10% 的 feature channel 設為 0，避免 overfitting

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                            # 將每個 feature map 設為 1 * 1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    
if __name__ == "__main__":
    model = CIFAR100CNNV2()

    x = torch.randn(128, 3, 32, 32)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )