import copy
import random
from time import time

import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.aggregation import fedavg
from src.federated_learning.client import Client
from src.metric import accuracy, watermark_detection_rate, one_hot_encoding
from src.model.convnet import Detector, ConvNet
from src.setting import DEVICE, NUM_WORKERS, PRCT_TO_SELECT, MAX_EPOCH_CLIENT
from src.model.freeze import bn_layers_requires_grad, embedding_mode_requies_grad
from src.plot import plot_FHE


class Server_Simulated_FHE():
    """
    The Server_FHE class represents a server in a federated learning system. The server manages the training process
    across multiple clients and embed the watermark in the encrypted global model.

    Attributes:
        model TODO
        dataset (str): The dataset used for training.
        nb_clients (int): The number of clients in the federated learning system.
        id (str): The identifier for the server.

    Methods:
        train(nb_rounds: int, lr_client: float, lr_pretrain: float, lr_retrain: float):
            Trains the model across multiple clients for a specified number of rounds. The learning rates for the
            clients, pretraining, and retraining can be specified.

        train_overwriting(nb_rounds: int, lr_client: float, lr_pretrain: float, lr_retrain: float, params: Tuple):
            TODO

        pretrain(lr_pretrain: float):
            TODO

        retrain(lr_retrain: float, max_round: int):
            TODO
    """

    def __init__(self, model: str, dataset: str, nb_clients: int, id: str):

        self.dataset = dataset
        self.nb_clients = nb_clients
        self.model = ConvNet(True)
        self.model.to(DEVICE)
        self.train_subsets, self.subset_size, self.test_set = data_splitter(self.dataset,
                                                                            self.nb_clients
                                                                            )
        self.detector = Detector()
        self.detector.to(DEVICE)
        self.model_test = ConvNet(False)
        self.model_test.to(DEVICE)
        self.trigger_set = torch.utils.data.DataLoader(
            WafflePattern(RGB=True, features=True),
            batch_size=10,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.id = id
        self.max_round = 30

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)

    def train(self, nb_rounds: int, lr_client: float, lr_pretrain: float, lr_retrain: float) -> None:
        print("#" * 60 + " Dynamic Watermarking for Encrypted Model " + "#" * 60)

        acc_test_list = []
        acc_watermark_black_list = []

        self.pretrain(lr_pretrain)

        print("Number of rounds :", nb_rounds)

        clients = []

        for c in range(self.nb_clients):
            client = Client("à changer", self.model.state_dict(), self.train_subsets[c])

            clients.append(client)

        for r in range(nb_rounds):

            print("")

            selected_clients = random.sample(
                range(self.nb_clients), int(PRCT_TO_SELECT * self.nb_clients)
            )

            loop = tqdm(selected_clients)

            for idx, c in enumerate(loop):
                clients[c].model.load_state_dict(self.model.state_dict())

                clients[c].train(lr=lr_client)

                loop.set_description(f"Round [{r}/{nb_rounds}]")

            fedavg(np.array(clients), self.model, self.subset_size, selected_clients)

            self.model_test.load_state_dict(self.model.state_dict())

            time_before = time()

            acc_watermark_black = self.retrain(lr_retrain, self.max_round)

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = accuracy(self.model_test, self.test_set)

            acc_test_list.append(acc_test)

            acc_watermark_black_list.append(acc_watermark_black)

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            lr_retrain = lr_retrain * 0.99

            plot_FHE(acc_test_list, acc_watermark_black_list, self.id)

        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

    def train_overwriting(self, nb_rounds, lr_client, lr_pretrain, lr_retrain, params):
        print("#" * 60 + "\tFHE\t" + "#" * 60)

        lr = lr_retrain

        acc_test_list = []
        acc_watermark_black_list = []

        print("Number of rounds :", nb_rounds)

        watermark_set, dynamic_key = params
        wdr_dynamic = []

        clients = []

        for c in range(self.nb_clients):
            client = Client(self.model.state_dict(), self.train_subsets[c], self.poly_client)

            clients.append(client)

        for r in range(nb_rounds):

            print("")

            selected_clients = random.sample(
                range(self.nb_clients), int(PRCT_TO_SELECT * self.nb_clients)
            )

            loop = tqdm(selected_clients)

            for idx, c in enumerate(loop):
                clients[c].model.load_state_dict(self.model.state_dict())

                clients[c].train(lr=lr_client)

                loop.set_description(f"Round [{r}/{nb_rounds}]")

            self.model = fedavg(np.array(clients), self.subset_size, self.model, selected_clients)

            self.model_test.load_state_dict(self.model.state_dict())

            time_before = time()

            acc_watermark_black = self.retrain(lr_retrain, max_round)

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = test(self.model_test, self.test_set)

            acc_test_list.append(acc_test)

            acc_watermark_black_list.append(acc_watermark_black)

            plot_FHE(
                acc_test_list, acc_watermark_black_list, acc_watermark_white_list, lr_client, lr_pretrain, lr,
                self.id
            )

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            lr_retrain = lr_retrain * 0.99

            dynamic_key_current = copy.deepcopy(self.detector)

            self.detector = dynamic_key

            _, tmp_dynamic, tmp_static = get_accuracies(self.model, self.test_set, watermark_set, key, message,
                                                        dynamic_key)

            self.detector = dynamic_key_current

            wdr_dynamic.append(tmp_dynamic)
            wdr_static.append(tmp_static)

            plot_overwriting(acc_test_list, acc_watermark_black_list, acc_watermark_white_list,
                             wdr_dynamic, wdr_static)

        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(self.secret_key, "./outputs/secret_key_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )

        torch.save(self.message, "./outputs/message_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )

    def pretrain(self, lr_pretrain: float) -> float:
        bn_layers_requires_grad(self.model, False)

        embedding_mode_requies_grad(self.model, False)

        optimizer = optim.SGD(self.model.fc1.parameters(), lr=lr_pretrain)

        optimizer_detector = optim.SGD(self.detector.parameters(), lr=1e-1)

        criterion = nn.MSELoss()

        acc_watermark_black, _ = watermark_detection_rate(self.model, self.detector, self.trigger_set)

        print("Black-Box WDR:", acc_watermark_black)

        print("\n" + "#" * 20 + " Watermark Pre-Embedding " + "#" * 20 + "\n")

        epoch = 0

        w0 = self.model.fc1.weight.data.detach().clone()

        while acc_watermark_black < 1.0:

            accumulate_loss = 0

            for inputs, outputs in self.trigger_set:
                optimizer.zero_grad(set_to_none=True)
                optimizer_detector.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                outputs = one_hot_encoding(outputs)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs, True)

                    outputs_predicted = self.detector(outputs_predicted)

                    diff = (1 / 2) * (w0 - self.model.fc1.weight).pow(2).sum()

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    loss = blackbox_loss + (1e-2 * diff)

                loss.backward()

                optimizer.step()
                optimizer_detector.step()

                accumulate_loss += loss.item()

            acc_watermark_black, _ = watermark_detection_rate(self.model, self.detector, self.trigger_set)

            print(f'\rBlack-Box WDR: {acc_watermark_black}, Loss: {round(accumulate_loss, 3)}', end='', flush=True)

            epoch += 1

        print("\n" + 60 * "#" + "\n")

        bn_layers_requires_grad(self.model, True)

        embedding_mode_requies_grad(self.model, True)

        return acc_watermark_black

    def retrain(self, lr_retrain: float, max_round: int) -> float:
        bn_layers_requires_grad(self.model, False)

        embedding_mode_requies_grad(self.model, False)

        print("\n" + "#" * 20 + " Watermark Re-Embedding " + "#" * 20 + "\n")

        optimizer = optim.SGD(self.model.fc1.parameters(), lr=lr_retrain)

        optimizer_detector = optim.SGD(self.detector.parameters(), lr=1e-2)

        criterion = nn.MSELoss()

        acc_watermark_black_before, loss_bb = watermark_detection_rate(self.model, self.detector, self.trigger_set)

        print("Black-Box WDR:", acc_watermark_black_before, loss_bb)

        loop = tqdm(list(range(max_round)))

        for idx, epoch in enumerate(loop):

            accumulate_loss = 0

            for inputs, outputs in self.trigger_set:
                optimizer.zero_grad(set_to_none=True)
                optimizer_detector.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                outputs = one_hot_encoding(outputs)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs, True)

                    outputs_predicted = self.detector(outputs_predicted)

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    regul = 1e-2 * torch.mean(self.model.fc1.weight, dim=0).pow(2).sum()

                    loss = blackbox_loss + regul

                loss.backward()

                optimizer.step()
                optimizer_detector.step()

                accumulate_loss += loss.item()

            acc_watermark_black, loss_watermark = watermark_detection_rate(self.model, self.detector, self.trigger_set)

            loop.set_description(f"Epoch [{epoch}/{max_round}]")

            loop.set_postfix(
                {
                    "Black-Box WDR :": acc_watermark_black,
                    "Watermarking Loss ": loss_watermark,
                }
            )

        print("\n" + 60 * "#" + "\n")

        bn_layers_requires_grad(self.model, True)

        embedding_mode_requies_grad(self.model, True)

        return acc_watermark_black_before
