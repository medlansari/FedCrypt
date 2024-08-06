import torch.nn as nn

from src.model.activation import Identity, ReLU_Poly


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, linear, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        self.linear = linear
        if linear:
            self.activation = Identity()
        else:
            self.activation = nn.GELU()

        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.bn1 = nn.BatchNorm2d(dim)
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            self.activation,
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    self.activation,
                    nn.BatchNorm2d(dim),
                )
                for _ in range(depth)
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(dim, 128),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        if self.linear:
            return x
        x = self.activation(x)
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


class convmixer_detector(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

        self.activation = ReLU_Poly()

    def forward(self, x):
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        z1 = self.fc1(x)
        z2 = self.activation(z1)
        return self.fc2(z2)
