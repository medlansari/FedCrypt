import unittest
import random

import numpy as np
import torch
from torch import nn

from src.data.data_splitter import data_splitter
from src.federated_learning.aggregation import fedavg
from src.federated_learning.client import Client
from src.model.convnet import ConvNet
from src.setting import DEVICE


def set_model_params_to_value(model : nn.Module, value : float) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)

def check_model_params_value(model, value):
    for param in model.parameters():
        if not torch.allclose(param, torch.tensor(value)):
            return False
    return True

class MyTestCase(unittest.TestCase):
    def test_two_clients(self):

        train_subsets, subset_size, test_set = data_splitter("MNIST",2)

        model = ConvNet(False)
        set_model_params_to_value(model, 32.0)

        c1 = Client("ConvNet", model.state_dict(), train_subsets[0])

        model = ConvNet(False)
        set_model_params_to_value(model, 48.0)

        c2 = Client("ConvNet", model.state_dict(), train_subsets[1])

        clients = np.array([c1, c2])

        model = ConvNet(False).to(DEVICE)
        set_model_params_to_value(model, 58.0)

        selected_clients = random.sample(
            range(2), int(1.0 * 2)
        )

        fedavg(clients, model, subset_size, selected_clients)

        all_close = check_model_params_value(model, (32+48)/2)

        self.assertEqual(True, all_close)  # add assertion here

    def test_five_clients(self):
        train_subsets, subset_size, test_set = data_splitter("MNIST", 5)

        model = ConvNet(False)
        set_model_params_to_value(model, 32.0)
        c1 = Client("ConvNet", model.state_dict(), train_subsets[0])

        model = ConvNet(False)
        set_model_params_to_value(model, 48.0)
        c2 = Client("ConvNet", model.state_dict(), train_subsets[1])

        model = ConvNet(False)
        set_model_params_to_value(model, 58.0)
        c3 = Client("ConvNet", model.state_dict(), train_subsets[2])

        model = ConvNet(False)
        set_model_params_to_value(model, 68.0)
        c4 = Client("ConvNet", model.state_dict(), train_subsets[3])

        model = ConvNet(False)
        set_model_params_to_value(model, 78.0)
        c5 = Client("ConvNet", model.state_dict(), train_subsets[4])

        clients = np.array([c1, c2, c3, c4, c5])

        model = ConvNet(False).to(DEVICE)
        set_model_params_to_value(model, 88.0)

        selected_clients = random.sample(
            range(5), int(1.0 * 5)
        )

        fedavg(clients, model, subset_size, selected_clients)

        all_close = check_model_params_value(model, (32 + 48 + 58 + 68 + 78) / 5)

        self.assertEqual(True, all_close)  # add assertion here

if __name__ == '__main__':
    unittest.main()
