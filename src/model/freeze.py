from torch import nn


def bn_layers_requires_grad(model, require):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(require)


def embedding_mode_requies_grad(model, bool):
    model.conv1.requires_grad_(bool)
    model.conv2.requires_grad_(bool)
    model.fc2.requires_grad_(bool)
    model.fc3.requires_grad_(bool)
