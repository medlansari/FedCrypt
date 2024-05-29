import numpy as np
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.client import Client
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.metric import accuracy, watermark_detection_rate
from src.model.vgg import Detector
from src.plot import plot_pruning_attack
from src.setting import NUM_WORKERS

path = "outputs"


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


def pruning(model, percentage_to_remove) -> None:
    parameters_to_prune = []

    for layer in get_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percentage_to_remove,
    )

    for layer in get_children(model):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            prune.remove(layer, 'weight')


def pruning_attack(ids: list[str]) -> None:

    train_subsets, subset_size, test_set = data_splitter("CIFAR10", 1)

    trigger_set = torch.utils.data.DataLoader(
        WafflePattern(RGB=True, features=True),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_accuracy = []
    wdr_dynamic = []

    for id in ids:

        pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for p in pruning_rates:
            Server = Server_Simulated_FHE("VGG", "CIFAR10", 10, id)

            Server.model.load_state_dict(torch.load(path + "/save_" + id + ".pth"))

            pruning(Server.model, p)

            Server.model_linear.load_state_dict(Server.model.state_dict())

            detector = Detector(10)
            detector.load_state_dict(torch.load(path + "/detector_" + id + ".pth"))
            detector.to("cuda")

            test_tmp = accuracy(Server.model, test_set)[0]
            dynamic_tmp, loss = watermark_detection_rate(Server.model_linear, detector, trigger_set)

            test_accuracy.append(test_tmp)
            wdr_dynamic.append(dynamic_tmp)

    test_accuracy = np.array(test_accuracy).reshape(-1, len(pruning_rates))
    wdr_dynamic = np.array(wdr_dynamic).reshape(-1, len(pruning_rates))

    print(test_accuracy)
    print(wdr_dynamic)

    plot_pruning_attack(pruning_rates, test_accuracy, wdr_dynamic)
