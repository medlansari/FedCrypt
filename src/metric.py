import torch
import torch.nn.functional as F
from torch import nn

from src.setting import DEVICE


def accuracy(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        total = 0

        correct = 0

        accumulate_loss = 0

        criterion = nn.CrossEntropyLoss()

        for inputs, outputs in loader:
            inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

            outputs = outputs.to(DEVICE)

            with torch.autocast(device_type="cuda"):
                outputs_predicted = model(inputs)

                loss = criterion(outputs_predicted, outputs)

            predicted = outputs_predicted.argmax(1)
            total += outputs.size(0)
            correct += predicted.eq(outputs).sum().item()

            accumulate_loss += loss.item()

    return round(correct / total, 3), round(accumulate_loss / len(loader), 3)


def one_hot_encoding(y: torch.Tensor) -> torch.Tensor:
    return (F.one_hot(y.to(torch.int64), num_classes=10)).float()


def watermark_detection_rate(
    model: nn.Module, detector: nn.Module, test_loader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    model.eval()
    detector.eval()

    with torch.no_grad():
        total = 0

        correct = 0

        accumulate_loss = 0

        criterion = nn.MSELoss()

        for inputs, outputs in test_loader:
            inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

            outputs = outputs.to(DEVICE)

            outputs = one_hot_encoding(outputs)

            with torch.autocast(device_type="cuda"):
                features_predicted = model(inputs)
                outputs_predicted = detector(features_predicted)

                loss = criterion(outputs_predicted, outputs)

            predicted = outputs_predicted.argmax(1)
            total += outputs.size(0)
            correct += predicted.eq(outputs.argmax(1)).sum().item()

            accumulate_loss += loss.item()

    return round(correct / total, 3), round(accumulate_loss / len(test_loader), 3)
