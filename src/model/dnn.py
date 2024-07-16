import torch
from torch import nn

from src.model.activation import Identity, ReLU_Poly


class DNN(torch.nn.Module):

    def __init__(self, n_features, n_classes, poly):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(400, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, n_classes)

        self.poly = poly
        if poly:
            self.activation = Identity
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x, features=False):
        z = self.bn1(self.conv1(x))
        z = self.pool(self.activation(z))

        z = self.bn2(self.conv2(z))
        z = self.pool(self.activation(z))

        z = torch.flatten(z, 1)

        z = self.activation(self.fc1(z))
        if features:
            return self.fc2(z)
        z = self.activation(self.fc2(z))
        y = self.fc3(z)

        return y


class Detector(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, n_classes)

        self.activation = ReLU_Poly()

    def forward(self, x):
        z = self.activation(self.fc1(x))
        return self.fc2(z)
