import numpy as np
import torch
from torch import optim, nn

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.client import Client
from src.metric import accuracy, watermark_detection_rate_white
from src.model.model_choice import model_choice
from src.setting import NUM_WORKERS, DEVICE

path = "./outputs"

def fine_tuning(method, model_name, dataset, id):
    match method:
        case "FedCrypt":
            fine_tuning_fedcrypt(model_name, dataset, 101, 1e-2 ,id)
        case "FedIPR":
            fine_tuning_white_box(model_name, dataset, 101, 1e-2 ,id)
        case "FedTracker":
            fine_tuning_white_box(model_name, dataset, 101, 1e-2 ,id)
        case _:
            raise NotImplementedError


def fine_tuning_fedcrypt(model_name, dataset, max_epoch, lr, id):
    train_subsets, subset_size, test_set, num_classes_task = data_splitter(
        dataset, 10
    )

    model, model_linear, detector = model_choice(model_name, 32 * 32, num_classes_task, 10)

    trigger_set = torch.utils.data.DataLoader(
        WafflePattern(RGB=True, features=False),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for i in range(1):
        client_malicious = Client(
            model_name,
            torch.load(path + "/save_" + id + ".pth"),
            32*32,
            num_classes_task,
            train_subsets[0],
        )

        detector.load_state_dict(torch.load(path + "/detector_" + id + ".pth"))
        detector.to("cuda")
        detector.eval()

        test_accuracy, wdr_dynamic = client_malicious.train_fine_tuning(
            lr, max_epoch, test_set, trigger_set, detector
        )

        np.savez(f"{path}/fine_tuning_{id}.pth", test_accuracy, wdr_dynamic)

def fine_tuning_white_box(
    model_name, dataset, max_epoch, lr, id
) -> tuple[list[float], list[float]]:

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

    optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    test_accuracy = []

    watermark_array = []

    acc_test, acc_loss = accuracy(model, test_set)

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

    watermark_array.append(acc_watermark)

    model.train()

    for epoch in range(max_epoch):
        accumulate_loss = 0

        for inputs, outputs in train_loader:
            optimizer.zero_grad(set_to_none=True)

            inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

            outputs = outputs.to(DEVICE)

            with torch.autocast(device_type="cuda"):
                outputs_predicted = model(inputs)

                loss = criterion(outputs_predicted, outputs)

            loss.backward()

            accumulate_loss += loss.item()

            optimizer.step()

        if epoch % 20 == 0:

            acc_test, acc_loss = accuracy(model, test_loader)

            test_accuracy.append(acc_test)

            acc_watermark, loss_watermark = watermark_detection_rate_white(
                model, secret_key, message
            )

            print(
                f"\rEpoch: {epoch}, Acc : {acc_test}, WDR : {acc_watermark}",
                end="",
                flush=True,
            )

            watermark_array.append(acc_watermark)

    np.savez(f"{path}/fine_tuning_{id}.pth", test_accuracy, watermark_array)
