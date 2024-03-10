import torch
import torch.nn as nn


class Yolo_v1(nn.Module):
    def __init__(self, bb_per_cell=2, grid_cells=7, num_classes=6):
        super().__init__()
        self.bb_per_cell = bb_per_cell
        self.grid_cells = grid_cells
        self.num_classes = num_classes
        self.darknet = nn.Sequential(
            # First batch
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),  # 448 x 448 -> 224 x 224
            nn.MaxPool2d(2, 2),  # 224 x 224 -> 112 x 112
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Second batch
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=3, padding=1
            ),  # 112 x 112 -> 112 x 112
            nn.MaxPool2d(2, 2),  # 112 x 112 -> 56 x 56
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Third batch
            nn.Conv2d(
                in_channels=192, out_channels=128, kernel_size=1, padding=0
            ),  # 56 x 56 -> 56 x 56
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # 56 x 56 -> 56 x 56
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=0
            ),  # 56 x 56 -> 56 x 56
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 56 x 56 -> 56 x 56
            nn.MaxPool2d(2, 2),  # 56 x 56 -> 28 x 28
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Fourth batch
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=1, padding=0
            ),  # 28 x 28 -> 28 x 28
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 28 x 28 -> 28 x 28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 x 28 -> 14 x 14
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Fith batch
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0
            ),  # 14 x 14 -> 14 x 14
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 14 x 14 -> 14 x 14
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0
            ),  # 14 x 14 -> 14 x 14
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 14 x 14 -> 14 x 14
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 14 x 14 -> 14 x 14
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1
            ),  # 14 x 14 -> 7 x 7
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Sixth batch
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 7 x 7 -> 7 x 7
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 7 x 7 -> 7 x 7
        )

        self.fc = nn.Sequential(
            nn.Linear(50176, 1000),  # 7 * 7 * 1024 = 50176
            nn.ELU(),
            nn.Dropout(0.45),
            nn.Linear(
                1000,
                self.grid_cells**2
                * (self.num_classes + 5 * self.bb_per_cell),  # <---
            ),
        )


    def forward(self, X):
        embed = self.get_embedding(X)
        out = self.fc(embed)
        return out

    def get_embedding(self, X):
        return self.darknet(X).flatten(1)
