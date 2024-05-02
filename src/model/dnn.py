import torch
from torch import nn

from src.model.activation import identity, relu_poly


class DNN(torch.nn.Module):

    def __init__(self, n_features, poly):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 16)
        self.fc2 = torch.nn.Linear(16, 2)

        self.poly = poly
        if poly:
            self.activation = identity
        else:
            self.activation = torch.nn.ReLU

    def forward(self, x, features):
        x = x.view(x.size(0), -1)
        if features:
            return relu_poly(self.fc1(x))
        z = self.activation(self.fc1(x))
        out = self.fc2(z)
        return out

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 2)

        self.activation = identity

    def forward(self, x):
        return self.fc1(x)