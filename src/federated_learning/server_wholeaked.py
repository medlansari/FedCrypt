import logging
import random
from copy import deepcopy
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
from src.metric import accuracy, watermark_detection_rate, one_hot_encoding, watermark_detection_rate_key
from src.model.model_choice import model_choice
from src.model.resnet import resnet_detector_feature
from src.plot import plot_FHE
from src.setting import DEVICE, NUM_WORKERS, PRCT_TO_SELECT, MAX_EPOCH_CLIENT


class Server_Wholeaked:

    def __init__(self, model: str, dataset: str, nb_clients: int, id: str):

        logger.log(logging.INFO, "Server Initialization")

        self.dataset = dataset
        self.nb_clients = nb_clients
        self.model_name = model
        self.num_classes_watermarking = 10
        self.input_size = 32 * 32

        self.train_subsets, self.subset_size, self.test_set, self.num_classes_task = data_splitter(
            self.dataset, self.nb_clients
        )

        self.model, _, _ = model_choice(
            self.model_name, self.input_size, self.num_classes_task, self.num_classes_watermarking
        )
        self.model.feature_extraction = True
        self.detector = resnet_detector_feature()
        self.model.to(DEVICE)
        self.detector.to(DEVICE)

        self.message = (torch.randint(2, (32,), device="cpu").float() - 0.5) * 2

        self.trigger_set = torch.utils.data.DataLoader(
            WafflePattern(RGB=True,features=True, message=self.message),
            batch_size=10,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.id = id
        self.max_round = 10

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

        # self.encrypted_pre_embedding(lr_pretrain)

        # for name, param in self.model.named_parameters():
        #    print(f"Layer: {name} | Trainable: {param.requires_grad}")

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

            acc_watermark_black = 0

            if r >= 3:
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

            # lr_client = lr_client * 0.99

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

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
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

        wdr_old, loss_old = watermark_detection_rate_key(
            self.model, original_detector, original_trigger_set
        )

        acc_watermark_org_list.append(wdr_old)

        print(
            "Old watermark detection rate: ", wdr_old, "Old watermark loss: ", loss_old
        )

        wdr_new, loss_new = watermark_detection_rate_key(
            self.model, self.detector, self.trigger_set
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
                self.model, original_detector, original_trigger_set
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

        acc_watermark_black, loss_bb = watermark_detection_rate_key(
            self.model, self.detector, self.trigger_set
        )

        print("Black-Box WDR:", acc_watermark_black, loss_bb)

        self.model.train()
        self.detector.train()

        optimizer = optim.SGD(
            self.model.parameters(), lr=lr_pretrain[0]
        )

        criterion = nn.CrossEntropyLoss()

        epoch = 0

        while acc_watermark_black < 1.0:

            accumulate_loss = 0

            for inputs, outputs in self.trigger_set:
                optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                with torch.autocast(device_type="cuda"):
                    features_predicted = self.model(inputs)

                    outputs_predicted = self.detector(features_predicted)

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    loss = blackbox_loss

                loss.backward()

                optimizer.step()

                accumulate_loss += loss.item()

            acc_watermark_black, loss_bb = watermark_detection_rate_key(
                self.model, self.detector, self.trigger_set
            )

            print(
                f"\rBlack-Box WDR: {acc_watermark_black}, Loss: {loss_bb}",
                end="",
                flush=True,
            )

            epoch += 1

            if epoch > 300:
                break

        # bn_layers_requires_grad(self.model, True)

        print("")

        logger.log(logging.WATERMARK, "Pre-Embedding Done")

        return acc_watermark_black

    def encrypted_re_embedding(self, lr_retrain: float, max_round: int) -> float:

        logger.log(logging.WATERMARK, "Re-Embedding")

        acc_watermark_black_before, loss_bb = watermark_detection_rate_key(
            self.model, self.detector, self.trigger_set
        )

        self.model.train()

        # Freeze all parameters of the detector
        for param in self.detector.parameters():
            param.requires_grad = False

        optimizer = optim.SGD(
            self.model.parameters(), lr=lr_retrain[0]
        )

        model_original = deepcopy(self.model)

        criterion = nn.CrossEntropyLoss()

        print("Black-Box WDR:", acc_watermark_black_before, loss_bb)

        loop = tqdm(list(range(max_round)))

        for idx, epoch in enumerate(loop):

            accumulate_loss = 0

            for inputs, outputs in self.trigger_set:
                optimizer.zero_grad(set_to_none=True)

                diff = 0

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                with torch.autocast(device_type="cuda"):
                    features_predicted = self.model(inputs)

                    outputs_predicted = self.detector(features_predicted)

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    for param, param_original in zip(self.model.parameters(), model_original.parameters()):
                        diff += torch.sum((param - param_original) ** 2)

                    loss = blackbox_loss + diff

                loss.backward()

                optimizer.step()

                accumulate_loss += loss.item()

            acc_watermark_black, loss_watermark = watermark_detection_rate_key(
                self.model, self.detector, self.trigger_set
            )

            loop.set_description(f"Epoch [{epoch}/{max_round}]")

            loop.set_postfix(
                {
                    "Black-Box WDR :": acc_watermark_black,
                    "Watermarking Loss ": loss_watermark,
                }
            )

        logger.log(logging.WATERMARK, "Re-Embedding Done")

        print(blackbox_loss, diff)

        return acc_watermark_black_before
