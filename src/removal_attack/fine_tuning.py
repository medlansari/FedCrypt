import numpy as np
import torch

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.client import Client
from src.model.convnet import Detector
from src.setting import NUM_WORKERS

path = "outputs"

def fine_tuning_attack(max_epoch, lr, id):
    train_subsets, subset_size, test_set = data_splitter("CIFAR10", 1)

    trigger_set =  torch.utils.data.DataLoader(
            WafflePattern(RGB=True, features=True),
            batch_size=10,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    for i in range(5):
        client_malicious = Client(torch.load(path+"/save_"+id+".pth"), train_subsets[0], poly=False)
        dynamic_key = Detector()
        dynamic_key.load_state_dict(torch.load(path+"/detector_"+id+".pth"))
        dynamic_key.to("cuda")

        test_accuracy, wdr_dynamic, wdr_static = client_malicious.train_fine_tuning(lr, max_epoch, dynamic_key, trigger_set)

        np.savez(
                path
                + "fine-tuning" + id + str(i),
                test_accuracy,
                wdr_dynamic
            )