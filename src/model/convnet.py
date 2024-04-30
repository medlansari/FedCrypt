import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.activation import relu_poly, identity


class ConvNet(nn.Module):
    def __init__(self, remove_activation):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.remove_activation = remove_activation

        if remove_activation:
            self.activation = identity
        else:
            self.activation = F.relu

    def forward(self, x, extract_features=False):
        z = self.bn1(self.conv1(x))
        z = self.pool(self.activation(z))

        z = self.bn2(self.conv2(z))
        z = self.pool(self.activation(z))

        z = torch.flatten(z, 1)

        z = self.activation(self.fc1(z))
        if extract_features:
            return z
        z = self.activation(self.fc2(z))
        y = self.fc3(z)

        return y


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(120, 64))

        self.fc2 = nn.Sequential(nn.Linear(64, 10))

        self.activation = relu_poly

    def forward(self, x):
        z = self.activation(self.fc1(x))
        return self.fc2(z)
