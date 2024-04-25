import torch


def relu_poly(x):
    return (0.09 * torch.pow(x, 2)) + (0.5 * x) + 0.47


def identity(x):
    return x
