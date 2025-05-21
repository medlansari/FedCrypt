import logging
import random
from time import time

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.logger import logger
from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.aggregation import fedavg
from src.federated_learning.client import Client
from src.metric import accuracy, watermark_detection_rate, one_hot_encoding, watermark_detection_rate_white
from src.model.model_choice import model_choice
from src.plot import plot_FHE
from src.setting import DEVICE, NUM_WORKERS, PRCT_TO_SELECT, MAX_EPOCH_CLIENT


class Server_FedTracker:
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

        logger.log(logging.INFO, "Server Initialization")

        self.dataset = dataset
        self.nb_clients = nb_clients
        self.model_name = model
        self.num_classes_watermarking = 100
        self.input_size = 32 * 32

        self.train_subsets, self.subset_size, self.test_set, self.num_classes_task = data_splitter(
            self.dataset, self.nb_clients
        )

        self.model, _, _ = model_choice(
            self.model_name, self.input_size, self.num_classes_task, self.num_classes_watermarking
        )
        self.model.to(DEVICE)

        self.secret_key = torch.randn((512, 256), device="cuda")
        self.message = (torch.randint(2, (256,), device="cuda").float() - 0.5) * 2

        self.id = id
        self.max_round = 30

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)

        logger.log(logging.INFO, "Server Initialization Done")
        print("")

    def train(
        self,
        nb_rounds: int,
        lr_client: float,
        lr_pretrain: (float, float),
        lr_retrain: (float, float),
    ) -> None:

        logger.log(logging.INFO, "FL Training")

        print("Number of rounds :", nb_rounds)

        acc_test_list = []
        acc_watermark_black_list = []

        self.encrypted_pre_embedding(lr_pretrain)

        clients = []

        for c in range(self.nb_clients):
            client = Client(
                self.model_name,
                self.model.state_dict(),
                self.input_size,
                self.num_classes_task,
                self.train_subsets[c],
            )

            clients.append(client)

        for r in range(nb_rounds):

            print("")

            logger.log(logging.CLIENT, "Training")

            selected_clients = random.sample(
                range(self.nb_clients), int(PRCT_TO_SELECT * self.nb_clients)
            )

            loop = tqdm(selected_clients)

            for idx, c in enumerate(loop):
                clients[c].model.load_state_dict(self.model.state_dict())

                clients[c].train(lr=lr_client)

                loop.set_description(f"Round [{r}/{nb_rounds}]")

            logger.log(logging.CLIENT, "Training Done")
            print("")

            fedavg(np.array(clients), self.model, self.subset_size, selected_clients)

            time_before = time()

            acc_watermark_black = self.encrypted_re_embedding(
                lr_retrain, self.max_round
            )

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = accuracy(self.model, self.test_set)

            acc_test_list.append(acc_test)

            acc_watermark_black_list.append(acc_watermark_black)

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            plot_FHE(acc_test_list, acc_watermark_black_list, self.id)



        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + self.model_name
            + "_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_"
            + self.id
            + ".pth",
        )


        logger.log(logging.INFO, "FL Training Done")

    def train_overwriting(
        self,
        original_trigger_set,
        original_detector,
        nb_rounds: int,
        lr_client: float,
        lr_pretrain: (float, float),
        lr_retrain: (float, float),
    ) -> None:

        print("Number of rounds :", nb_rounds)

        print("#" * 60 + " Dynamic Watermarking for Encrypted Model " + "#" * 60)

        acc_test_list = []
        acc_watermark_org_list = []
        acc_watermark_new_list = []

        wdr_old, loss_old = watermark_detection_rate(
            self.model_linear, original_detector, original_trigger_set
        )

        acc_watermark_org_list.append(wdr_old)

        print(
            "Old watermark detection rate: ", wdr_old, "Old watermark loss: ", loss_old
        )

        wdr_new, loss_new = watermark_detection_rate(
            self.model_linear, self.detector, self.trigger_set
        )

        acc_watermark_new_list.append(wdr_new)

        print(
            "New watermark detection rate: ", wdr_new, "New watermark loss: ", loss_old
        )

        acc_test_list.append(accuracy(self.model, self.test_set)[0])

        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Trainable: {param.requires_grad}")

        clients = []

        for c in range(self.nb_clients):
            client = Client(
                self.model_name,
                self.model.state_dict(),
                self.input_size,
                self.num_classes_task,
                self.train_subsets[c],
            )

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

            time_before = time()

            wdr_old, loss_old = watermark_detection_rate(
                self.model_linear, original_detector, original_trigger_set
            )

            acc_watermark_org_list.append(wdr_old)

            print("Original Black-Box WDR:", wdr_old, loss_old)

            wdr_new = self.encrypted_re_embedding(lr_retrain, self.max_round)

            acc_watermark_new_list.append(wdr_new)

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = accuracy(self.model, self.test_set)

            acc_test_list.append(acc_test)

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            # lr_retrain = lr_retrain * 0.99

            lr_client = lr_client * 0.99

        np.savez(
            "./outputs/save_" + "FHE_overwriting" + "_" + self.id + "_" + str(time()),
            acc_test_list,
            acc_watermark_org_list,
            acc_watermark_new_list,
        )

        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_"
            + self.id
            + ".pth",
        )

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_"
            + self.id
            + ".pth",
        )

    def encrypted_pre_embedding(self, lr_pretrain: float) -> float:

        logger.log(logging.WATERMARK, "Pre-Embedding")

        acc_watermark_white, loss_wb = watermark_detection_rate_white(self.model, self.secret_key, self.message)

        print(f"\rWhite-Box WSR: {acc_watermark_white}, Loss: {loss_wb}")

        optimizer = optim.SGD(
            self.model.classifier[4].parameters(), lr=lr_pretrain[0]
        )

        criterion = lambda x : torch.sum(torch.relu(1 - (x * self.message)))

        epoch = 0

        while acc_watermark_white < 1.0:

            accumulate_loss = 0

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda"):

                reconstructed_message = self.model.classifier[4].weight.mean(0) @ self.secret_key

                loss = criterion(reconstructed_message)

                loss.backward()

                optimizer.step()

                accumulate_loss += loss.item()

            acc_watermark_white, loss_wb = watermark_detection_rate_white(self.model, self.secret_key, self.message)

            print(
                f"\rWhite-Box WSR: {acc_watermark_white}, Loss: {loss_wb}",
                end="",
                flush=True,
            )

            epoch += 1

            if epoch > 300:
                break

        print("")

        logger.log(logging.WATERMARK, "Pre-Embedding Done")

        return acc_watermark_white

    def encrypted_re_embedding(self, lr_retrain: float, max_round: int) -> float:

        logger.log(logging.WATERMARK, "Re-Embedding")

        acc_watermark_white, loss_wb = watermark_detection_rate_white(self.model, self.secret_key, self.message)

        print(f"\rWhite-Box WSR: {acc_watermark_white}, Loss: {loss_wb}")

        optimizer = optim.SGD(
            self.model.classifier[4].parameters(), lr=lr_retrain[0]
        )

        criterion = lambda x: torch.sum(torch.relu(1 - (x * self.message)))

        for i in range(max_round):

            accumulate_loss = 0

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda"):

                reconstructed_message = self.model.classifier[4].weight.mean(0) @ self.secret_key

                loss = criterion(reconstructed_message)

                loss.backward()

                optimizer.step()

                accumulate_loss += loss.item()

            acc_watermark_white, loss_wb = watermark_detection_rate_white(self.model, self.secret_key, self.message)

            print(
                f"\rWhite-Box WSR: {acc_watermark_white}, Loss: {loss_wb}",
                end="",
                flush=True,
            )

        print("")

        logger.log(logging.WATERMARK, "Re-Embedding Done")

        return acc_watermark_white
