import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def estimate_params(c1, c2, fc1, fc2, fc3):
    conv1_params = (3 * 5 * 5 + 1) * c1
    conv2_params = (c1 * 5 * 5 + 1) * c2
    fc1_params = (c2 * 5 * 5 + 1) * fc1
    fc2_params = (fc1 + 1) * fc2
    fc3_params = (fc2 + 1) * fc3
    return conv1_params + conv2_params + fc1_params + fc2_params + fc3_params

class Net(nn.Module):
    def __init__(self, target_params=None):
        super().__init__()

        # Default values
        c1, c2 = 6, 16
        fc1, fc2, fc3 = 120, 84, 10

        if target_params:
            # Adjust parameters iteratively to fit within target
            scale = (target_params / estimate_params(c1, c2, fc1, fc2, fc3)) ** 0.5
            c1, c2 = int(c1 * scale), int(c2 * scale)
            fc1, fc2 = int(fc1 * scale), int(fc2 * scale)

            # Recalculate to get final parameter count
            final_params = estimate_params(c1, c2, fc1, fc2, fc3)
            assert abs(final_params - target_params) / target_params <= 0.05, "Final params exceed 5% tolerance"

        self.conv1 = nn.Conv2d(3, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(c2 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv1Segment(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x


class Conv2Segment(nn.Module):
    def __init__(self, in_channels=6, out_channels=16):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        return x


class FC1Segment(nn.Module):
    def __init__(self, in_features=16 * 5 * 5, out_features=120):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Flatten before the FC layer
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x


class FC2Segment(nn.Module):
    def __init__(self, in_features=120, out_features=84):
        super().__init__()
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return x


class FC3Segment(nn.Module):
    def __init__(self, in_features=84, out_features=10):
        super().__init__()
        self.fc3 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc3(x)
