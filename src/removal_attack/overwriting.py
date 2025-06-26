from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn
import random

from src.data.data_splitter import data_splitter
from src.data.trigger_pgd import PGDSet
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.metric import accuracy, watermark_detection_rate_white, watermark_criterion
from src.model.model_choice import model_choice
from src.setting import NUM_WORKERS, DEVICE

path = "./outputs"

def overwriting(method, model_name, dataset, id):
    match method:
        case "FedCrypt":
            overwriting_fedcrypt(model_name, dataset, 101, id)
        case "FedIPR":
            overwriting_white_box(model_name, dataset,id)
        case "FedTracker":
            overwriting_white_box(model_name, dataset,id)
        case _:
            raise NotImplementedError

def overwriting_fedcrypt(model_name, dataset, epoch, id):
    train_subsets, subset_size, test_set, num_classes_task = data_splitter(
        dataset, 10
    )

    model, model_linear, detector = model_choice(model_name, 32 * 32, num_classes_task, 10)
    model.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model.to(DEVICE)

    model_linear.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model_linear.to(DEVICE)

    detector.load_state_dict(torch.load(f"{path}/detector_{id}.pth"))
    detector.to(DEVICE)

    original_watermark_set = torch.utils.data.DataLoader(
        WafflePattern(),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    new_watermark_set = torch.utils.data.DataLoader(
        PGDSet(),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    original_detector = deepcopy(detector)

    Server = Server_Simulated_FHE(model_name, dataset, 10, id)

    Server.trigger_set = new_watermark_set

    Server.model = deepcopy(model)
    Server.model_linear = deepcopy(model_linear)

    if dataset == "CIFAR100":

        Server.train_overwriting(
            original_watermark_set,
            original_detector,
            epoch,
            1e-3,
            (1e-3, 1e-2),
            (1e-3, 1e-2),
        )

    else:

        Server.train_overwriting(
            original_watermark_set,
            original_detector,
            epoch,
            1e-3,
            (1e-2, 1e-1),
            (1e-2, 1e-1),
        )





def overwriting_white_box(
    model_name: str,
    dataset: str,
id: str) -> None:

    train_subsets, subset_size, test_set, num_classes_task = data_splitter(
        dataset, 10
    )

    train_loader = train_subsets[0]
    test_loader = test_set

    model, _, _ = model_choice(model_name, 32 * 32, num_classes_task, num_classes_task)
    model.load_state_dict(torch.load(f"{path}/save_{id}.pth"))
    model.to(DEVICE)

    torch.manual_seed(0)
    dim_key = model.classifier[4].weight.shape[1]
    secret_key = torch.randn((dim_key, 256), device="cuda")
    message = (torch.randint(2, (256,), device="cuda").float() - 0.5) * 2

    seed = random.randint(0, 2 ** 32 - 1)
    torch.manual_seed(seed)
    # secret_key_attack = torch.randint(-2,2, (dim_key, 256), device="cuda").float()
    secret_key_attack = torch.normal(mean=0, std=0.15, size=(dim_key, 256),device="cuda")
    message_attack = (torch.randint(2, (256,), device="cuda").float() - 0.5) * 2

    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    criterion_watermarking = watermark_criterion

    test_accuracy = []

    wsr = []
    wsr_attack = []

    acc_test, acc_loss = accuracy(model, test_loader)

    test_accuracy.append(acc_test)

    acc_watermark, loss_watermark = watermark_detection_rate_white(
        model, secret_key, message
    )

    print(
        "Initial watermark detection rate: ",
        acc_watermark,
        "Initial watermark loss: ",
        loss_watermark,
    )

    acc_watermark_attack, loss_watermark_attack = watermark_detection_rate_white(
        model, secret_key_attack, message_attack
    )

    print(
        "Initial watermark detection rate for attacker: ",
        acc_watermark_attack,
        "Initial watermark loss for attacker: ",
        loss_watermark_attack,
    )

    print(" ")

    wsr.append(acc_watermark)
    wsr_attack.append(acc_watermark_attack)

    model.train()

    for epoch in range(101):
        accumulate_loss = 0

        for inputs, outputs in train_loader:
            optimizer.zero_grad(set_to_none=True)

            inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

            outputs = outputs.to(DEVICE)

            with torch.autocast(device_type="cuda"):
                outputs_predicted = model(inputs)

                reconstructed_message = model.classifier[4].weight.mean(0) @ secret_key_attack

                loss = criterion(outputs_predicted, outputs) + (1e2 * (criterion_watermarking(reconstructed_message, message_attack)))

            loss.backward()

            accumulate_loss += loss.item()

            optimizer.step()

        if epoch % 10 == 0 and epoch != 0:

            acc_test, acc_loss = accuracy(model, test_loader)

            test_accuracy.append(acc_test)

            acc_watermark, loss_watermark = watermark_detection_rate_white(
                model, secret_key, message
            )

            acc_watermark_attack, loss_watermark_attack = watermark_detection_rate_white(
                model, secret_key_attack, message_attack
            )

            print(
                f"\rEpoch: {epoch}, Acc : {acc_test}, WSR : {acc_watermark}, WSR Attack : {acc_watermark_attack}",
                end="",
                flush=True,
            )

            wsr.append(acc_watermark)
            wsr_attack.append(acc_watermark_attack)

    np.savez(f"{path}/overwriting_{id}", test_accuracy, wsr, wsr_attack)
