from torch import nn


def bn_layers_requires_grad(model: nn.Module, require: bool) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(require)


def embedding_mode_requies_grad(model: nn.Module, require: bool) -> None:
    model.conv1.requires_grad_(require)
    model.conv2.requires_grad_(require)
    model.fc2.requires_grad_(require)
    model.fc3.requires_grad_(require)
