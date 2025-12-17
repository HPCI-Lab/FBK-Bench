
import torch.nn as nn

class LargeMNISTCNN(nn.Module):
    def __init__(self, width=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),

            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),

            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((width * 2) * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
