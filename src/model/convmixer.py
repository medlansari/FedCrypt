import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torch.nn as nn

from src.model.activation import Identity


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, linear, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        if linear:
            self.activation = Identity()
        else:
            self.activation = nn.GELU()

        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.bn1 = nn.BatchNorm2d(dim)
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    self.activation,
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                self.activation,
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def trainable(self):
        # Geler tous les paramètres
        for param in self.parameters():
            param.requires_grad = False

        # Décongeler les paramètres de la dernière couche
        for param in self.classifier.parameters():
            param.requires_grad = True

def convmixer(linear=False):
    if linear:
        return ConvMixer(256, 8, True, 5, 2, 10)
    else:
        return ConvMixer(256, 8, False, 5, 2, 10)