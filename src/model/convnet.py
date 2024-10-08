import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.activation import ReLU_Poly, Identity


class ConvNet(nn.Module):
    def __init__(self, linear, out_layer):
        super().__init__()

        self.linear = linear
        self.outlayer = out_layer

        if self.linear:
            self.activation = Identity
        else:
            self.activation = F.relu

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        z = self.bn1(self.conv1(x))
        z = self.pool(self.activation(z))

        if self.linear and self.outlayer == 0:
            return torch.flatten(z, 1)

        z = self.bn2(self.conv2(z))
        z = self.pool(self.activation(z))

        z = torch.flatten(z, 1)

        if self.linear and self.outlayer == 1:
            return z

        z = self.activation(self.fc1(z))

        if self.linear and self.outlayer == 2:
            return torch.flatten(z, 1)

        z = self.activation(self.fc2(z))

        if self.linear and self.outlayer == 3:
            return torch.flatten(z, 1)

        y = self.fc3(z)

        return y


class Detector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 32))

        self.fc2 = nn.Sequential(nn.Linear(32, 10))

        self.activation = ReLU_Poly()

    def forward(self, x):
        z = self.activation(self.fc1(x))
        return self.fc2(z)
