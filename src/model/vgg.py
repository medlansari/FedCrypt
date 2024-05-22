from torchvision import models
import torch
from torch.functional import F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from src.model.activation import ReLU_Poly, Identity


# class VGG11(nn.Module):
#     def __init__(self, n_classes=10):
#         super(VGG11, self).__init__()
#         self.model = models.vgg11(weights=None)
#         self.model.classifier[-1] = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=n_classes))
#
#         def replace_maxpool_with_avgpool(model):
#             for name, module in model.named_children():
#                 if isinstance(module, nn.MaxPool2d):
#                     setattr(model, name, nn.AvgPool2d(module.kernel_size, module.stride, module.padding))
#                 else:
#                     replace_maxpool_with_avgpool(module)
#
#         replace_maxpool_with_avgpool(self.model)
#
#         self.extract_features = create_feature_extractor(self.model, return_nodes={"classifier.6.0" : "target"})
#
#     def forward(self, x, feature=False):
#         if feature:
#             return self.extract_features(x)["target"]
#         else:
#             return self.model(x)
#
#     def freeze(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for param in self.model.classifier[-1][0].parameters():
#             param.requires_grad = True
#
#         self.replace_activations(self.model, nn.ReLU, Identity())
#
#     def unfreeze(self):
#         for param in self.model.parameters():
#             param.requires_grad = True
#
#         self.replace_activations(self.model, Identity, nn.ReLU())
#
#     def targeted_layer(self):
#         return self.model.classifier[-1][0]
#
#
#     def replace_activations(self, model, old_activation, new_activation):
#         for name, module in model.named_children():
#             if isinstance(module, old_activation):
#                 setattr(model, name, new_activation)
#             else:
#                 self.replace_activations(module, old_activation, new_activation)

class VGG11(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # fully connected linear layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

def create_vgg11(n_classes=10):
    model = VGG11(in_channels=3, num_classes=n_classes)
    return model

def replace_maxpool_with_avgpool(model):
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool2d):
            setattr(model, name, nn.AvgPool2d(module.kernel_size, module.stride, module.padding))
        elif isinstance(module, nn.AdaptiveMaxPool2d):
            setattr(model, name, nn.AdaptiveAvgPool2d(module.output_size))
        else:
            replace_maxpool_with_avgpool(module)

def replace_avgpool_with_maxpool(model):
    for name, module in model.named_children():
        if isinstance(module, nn.AvgPool2d):
            setattr(model, name, nn.MaxPool2d(module.kernel_size, module.stride, module.padding))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            setattr(model, name, nn.AdaptiveMaxPool2d(module.output_size))
        else:
            replace_avgpool_with_maxpool(module)

def ext_features(model, x, feature=False):
    if feature:
        extract_features = create_feature_extractor(model, return_nodes={"classifier.0": "target"})
        return extract_features(x)["target"]
    else:
        return model(x)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[0].parameters():
        param.requires_grad = True
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.training = False
    replace_activations(model, nn.ReLU, Identity())
    replace_maxpool_with_avgpool(model)

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    replace_activations(model, Identity, nn.ReLU())
    replace_avgpool_with_maxpool(model)

def targeted_layer(model):
    return model.classifier[0]

def replace_activations(model, old_activation, new_activation):
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation)
        else:
            replace_activations(module, old_activation, new_activation)

class Detector(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(4096, 64)
        self.fc2 = nn.Linear(64, n_classes)

        self.activation = ReLU_Poly()

    def forward(self, x):
        z = self.activation(self.fc1(x))
        return self.fc2(z)
