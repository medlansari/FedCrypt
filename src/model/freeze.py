def bn_layers_requires_grad(self, require):
    for module in self.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(require)

def embedding_mode_requies_grad(self, bool):
    self.model.conv1.requires_grad_(bool)
    self.model.conv2.requires_grad_(bool)
    # self.model.fc1.requires_grad_(bool)
    self.model.fc2.requires_grad_(bool)
    self.model.fc3.requires_grad_(bool)