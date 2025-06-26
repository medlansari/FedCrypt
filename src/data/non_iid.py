import random
from collections import defaultdict

import numpy as np
import torch

from src.setting import BATCH_SIZE_CLIENT, NUM_WORKERS

class TensorDatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, self.target[index]


def distribute_non_iid_data(train_data, num_clients, alpha, transform):
    subdataset_indexes = distribute_per_class(train_data, num_clients, alpha)
    subdatasets = []
    for client in range(num_clients):
        selected = subdataset_indexes[client]
        data_selected = torch.tensor(train_data.data)[selected]
        data_selected = data_selected
        label_selected = torch.tensor(train_data.targets)[selected]
        tmp = TensorDatasetWithTransform(data_selected.float(), label_selected, transform)
        subdatasets.append(torch.utils.data.DataLoader(
            tmp,
            batch_size=BATCH_SIZE_CLIENT,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        ))
    print(f"Distributed {sum([len(loader.dataset) for loader in subdatasets])} samples among clients")
    print([len(loader.dataset) for loader in subdatasets])
    return subdatasets, [len(loader.dataset) for loader in subdatasets]


def distribute_per_class(train_data, num_clients, alpha):
    classes = {}
    # Get data indexes per label
    for ind, x in enumerate(train_data):
        _, label = x
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]
    # Get real data size
    class_size = len(classes[0])
    no_classes = len(classes.keys())
    # Fill client sample index list
    per_participant_list = defaultdict(list)
    for n in range(no_classes):
        random.shuffle(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(num_clients * [alpha]))
        for user in range(num_clients):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][
                           :min(len(classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            classes[n] = classes[n][
                         min(len(classes[n]), no_imgs):]
    return per_participant_list


