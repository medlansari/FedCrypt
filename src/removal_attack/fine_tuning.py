import numpy as np
import torch

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.client import Client
from src.model.vgg import Detector
from src.setting import NUM_WORKERS

path = "outputs"


def fine_tuning_attack(model_name, input_size, num_classes_task, max_epoch, lr, id):
    train_subsets, subset_size, test_set = data_splitter("CIFAR10", 1)

    trigger_set = torch.utils.data.DataLoader(
        WafflePattern(RGB=True, features=True),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for i in range(5):
        client_malicious = Client(model_name, torch.load(path + "/save_" + id + ".pth"), input_size, num_classes_task,
                                  train_subsets[0])
        detector = Detector(num_classes_task)
        detector.load_state_dict(torch.load(path + "/detector_" + id + ".pth"))
        detector.to("cuda")

        test_accuracy, wdr_dynamic = client_malicious.train_fine_tuning(lr, max_epoch, test_set, trigger_set, detector)

        np.savez(
            path
            + "fine-tuning" + id + str(i),
            test_accuracy,
            wdr_dynamic
        )
