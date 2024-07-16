from torchvision import models
import torch
from torch import nn


def init_vgg(n_classes=10):
    model = models.vgg11(weights=None)
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features=4096, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=n_classes))
    return model


def load_vgg(path, n_classes=10):
    model = models.vgg11(weights=None)
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features=4096, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=n_classes))
    model.load_state_dict(torch.load(path))

    return model
