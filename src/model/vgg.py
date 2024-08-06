import math

import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

from src.model.activation import ReLU_Poly, Identity


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, features, linear):
        super(VGG, self).__init__()
        self.features = features
        self.linear = linear
        if linear:
            self.classifier = nn.Sequential(
                Identity(),
                nn.Linear(512, 512),
                # nn.BatchNorm1d(512),
                Identity(),
                Identity(),
                nn.Linear(512, 512),
                # nn.BatchNorm1d(512),
                Identity(),
            )

            self.trainable()

        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(True),
            )
        self.last_layer = nn.Linear(512, 10)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if not (self.linear):
            return self.last_layer(x)
        return x

    def trainable(self):
        # Geler tous les paramètres
        for param in self.parameters():
            param.requires_grad = False

        # Décongeler les paramètres de la dernière couche
        for param in self.classifier[4].parameters():
            param.requires_grad = True


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_linear(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), Identity()]
            else:
                layers += [conv2d, Identity()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg11(linear=False):
    """VGG 11-layer model (configuration "A")"""
    if linear:
        return VGG(make_layers_linear(cfg["A"]), linear)
    else:
        return VGG(make_layers(cfg["A"]), linear)


def vgg11_bn(linear=False):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    if linear:
        return VGG(make_layers_linear(cfg["A"], batch_norm=True), linear)
    else:
        return VGG(make_layers(cfg["A"], batch_norm=True), linear)


class vgg_detector(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.activation = ReLU_Poly()

    def forward(self, x):
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        z1 = self.fc1(x)
        z2 = self.activation(z1)
        return self.fc2(z2)
