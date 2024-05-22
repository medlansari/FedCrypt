import torch

class ReLU_Poly(torch.nn.Module):
    def __init__(self):
        super(ReLU_Poly, self).__init__()
    def forward(self, x):
        return (0.09 * torch.pow(x, 2)) + (0.5 * x) + 0.47

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
