from copy import deepcopy

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

                loss = criterion(outputs_predicted, outputs.long())

            predicted = outputs_predicted.argmax(1)
            total += outputs.size(0)
            correct += predicted.eq(outputs).sum().item()

            accumulate_loss += loss.item()

    return round(correct / total, 4), round(accumulate_loss / len(loader), 4)


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

    return round(correct / total, 4), round(accumulate_loss / len(test_loader), 4)

def watermark_detection_rate_black(
    model: nn.Module, detector: nn.Module, test_loader: torch.utils.data.DataLoader, feature_extraction: bool = False
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
                features_predicted = model(inputs, features_extraction=feature_extraction)
                outputs_predicted = detector(features_predicted)

                loss = criterion(outputs_predicted, outputs)

            predicted = outputs_predicted.argmax(1)
            total += outputs.size(0)
            correct += predicted.eq(outputs.argmax(1)).sum().item()

            accumulate_loss += loss.item()

    return round(correct / total, 4), round(accumulate_loss / len(test_loader), 4)


def watermark_detection_rate_white(
    model: nn.Module, secret_key : torch.tensor, message : torch.tensor
) -> tuple[float, float]:

    reconstructed_message = model.classifier[4].weight.mean(0) @ secret_key

    reconstructed_message = torch.where(reconstructed_message >= 0, 1, -1)

    reconstructed_message = reconstructed_message.detach().cpu()

    return 1-((reconstructed_message != message.cpu()).sum()/message.size(0)).item(), 0

def watermark_detection_rate_key(
    model: nn.Module, detector: nn.Module, test_loader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    model.eval()
    detector.eval()

    with torch.no_grad():
        total = 0

        cumulated_ber = 0

        accumulate_loss = 0

        criterion = nn.CrossEntropyLoss()

        for inputs, outputs in test_loader:
            inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

            outputs = outputs.to(DEVICE)

            with torch.autocast(device_type="cuda"):
                features_predicted = model(inputs)
                outputs_predicted = detector(features_predicted)

            reconstructed_message = torch.where(outputs_predicted >= 0, 1, -1)

            reconstructed_message = reconstructed_message.detach().cpu()

            ber = torch.where(reconstructed_message + outputs.cpu() == 0,1.,0.).sum(1).mean()

            total += 1

            cumulated_ber += ber.item()

    return round(1-(cumulated_ber/(total*32)), 3), round(accumulate_loss / len(test_loader), 3)

def watermark_detection_rate_classhidden(
    model: nn.Module, watermark_layer: nn.Parameter, loader: torch.utils.data.DataLoader
) -> tuple[float, float]:

    model = deepcopy(model)

    model.last_layer = watermark_layer

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

                loss = criterion(outputs_predicted, outputs.long())

            predicted = outputs_predicted.argmax(1)
            total += outputs.size(0)
            correct += predicted.eq(outputs).sum().item()

            accumulate_loss += loss.item()

    return round(correct / total, 4), round(accumulate_loss / len(loader), 4)


def watermark_criterion(reconstructed_message: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.relu(1 - (reconstructed_message * message)))