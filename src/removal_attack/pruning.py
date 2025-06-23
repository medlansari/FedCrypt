from copy import deepcopy

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch import nn

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.metric import accuracy, watermark_detection_rate, watermark_detection_rate_white
from src.model.model_choice import model_choice
from src.plot import plot_pruning_attack
from src.setting import NUM_WORKERS, DEVICE

path = "outputs"

def pruning(method, model_name, dataset, id):
    match method:
        case "FedCrypt":
            pruning_fedcrypt(model_name, dataset, id)
        case "FedIPR":
            pruning_white_box(model_name, dataset,id)
        case "FedTracker":
            pruning_white_box(model_name, dataset, id)
        case _:
            raise NotImplementedError


def get_children(model: torch.nn.Module) -> list[torch.nn.Module]:
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def apply_pruning(model, percentage_to_remove) -> None:
    parameters_to_prune = []

    for layer in get_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percentage_to_remove,
    )

    for layer in get_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            prune.remove(layer, "weight")


def pruning_fedcrypt(model_name, dataset, id) -> None:
    train_subsets, subset_size, test_set, num_classes_task = data_splitter(
        dataset, 10
    )

    model, model_linear, detector = model_choice(model_name, 32 * 32, num_classes_task, num_classes_task)
    model.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model.to(DEVICE)

    model_linear.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model_linear.to(DEVICE)

    detector.load_state_dict(torch.load(f"{path}/detector_{id}.pth"))
    detector.to(DEVICE)


    trigger_set = torch.utils.data.DataLoader(
        WafflePattern(RGB=True, features=False),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    acc_watermark, loss_watermark = watermark_detection_rate(
        model_linear, detector, trigger_set
    )

    print(
        "Initial watermark detection rate: ",
        acc_watermark,
        "Initial watermark loss: ",
        loss_watermark,
    )

    test_accuracy = []
    wsr = []

    pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    original_model = deepcopy(model)

    for p in pruning_rates:
        model.load_state_dict(original_model.state_dict())

        apply_pruning(model, p)

        model_linear.load_state_dict(model.state_dict())

        test_accuracy.append(accuracy(model, test_set))
        wsr_acc, wsr_loss = watermark_detection_rate(model_linear, detector, trigger_set)

        wsr.append((wsr_acc,wsr_loss))

    np.savez(f"{path}/pruning_{id}.pth", test_accuracy, wsr)


def pruning_white_box(
    model_name: str,
    dataset: str,
    id: str
) -> tuple[list[float], list[float]]:

    train_subsets, subset_size, test_set, num_classes_task = data_splitter(
            dataset, 10
    )

    model, _, _ = model_choice(model_name, 32 * 32, num_classes_task, num_classes_task)
    model.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model.to(DEVICE)

    torch.manual_seed(0)
    dim_key = model.classifier[4].weight.shape[1]
    secret_key = torch.randn((dim_key, 256), device="cuda")
    message = (torch.randint(2, (256,), device="cuda").float() - 0.5) * 2

    acc_watermark, loss_watermark = watermark_detection_rate_white(
        model, secret_key, message
    )

    print(
        "Initial watermark detection rate: ",
        acc_watermark,
        "Initial watermark loss: ",
        loss_watermark,
    )

    test_accuracy = []
    wdr = []

    pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    original_model = deepcopy(model)

    for p in pruning_rates:

        model.load_state_dict(original_model.state_dict())

        apply_pruning(model, p)

        test_accuracy.append(accuracy(model, test_set))
        wdr.append(watermark_detection_rate_white(model, secret_key, message))

    np.savez(f"{path}/pruning_{id}.pth", test_accuracy, wdr)
