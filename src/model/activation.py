import torch


def relu_poly(x: torch.Tensor) -> torch.Tensor:
    return (0.09 * torch.pow(x, 2)) + (0.5 * x) + 0.47


def identity(x: torch.Tensor) -> torch.Tensor:
    return x
