import random
from time import time

import sklearn
import tenseal as ts

import numpy as np
import torch
from tqdm import tqdm

from src.data.fake_data import RandomTriggerSet
from src.federated_learning.aggregation import fedavg
from src.federated_learning.client import Client
from src.metric import accuracy
from src.model.dnn import DNN, Detector
from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.model.encrypted_model import EncryptedModel
from src.model.model_choice import model_choice
from src.plot import plot_FHE
from src.setting import NUM_WORKERS, DEVICE, PRCT_TO_SELECT, MAX_EPOCH_CLIENT


class Server_Real_FHE:
    def __init__(self, model: str, dataset: str, nb_clients: int, id: str):
        self.model_name = model
        self.dataset = dataset
        self.nb_clients = nb_clients
        self.num_classes_task = 10
        self.num_classes_watermarking = 2
        self.input_size = 32 * 32

        self.model = model_choice(self.model_name, self.input_size, self.num_classes_task)
        self.model.to(DEVICE)

        self.train_subsets, self.subset_size, self.test_set = data_splitter(self.dataset, self.nb_clients)
        self.detector = Detector(self.num_classes_watermarking)
        self.detector.to(DEVICE)

        self.id = id

        # parameters
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        # create TenSEALContext
        self.ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        self.ctx_training.global_scale = 2 ** 21
        self.ctx_training.generate_galois_keys()

        self.trigger_set = RandomTriggerSet(self.ctx_training, 10, 64, self.num_classes_watermarking)
        self.model_encrypted = EncryptedModel(self.model, Detector(self.num_classes_watermarking), 2, self.ctx_training)

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)

    def train(self, nb_rounds: int, lr_client: float, epoch_pretrain: int, epoch_retrain: int) -> None:
        print("#" * 60 + " Dynamic Watermarking for Encrypted Model " + "#" * 60)

        acc_test_list = []
        acc_watermark_black_list = []

        # self.encrypted_pre_embedding(epoch_pretrain)

        print("Number of rounds :", nb_rounds)

        clients = []

        for c in range(self.nb_clients):
            client = Client(self.model_name, self.model.state_dict(),
                            self.input_size, self.num_classes_task, self.train_subsets[c])

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

            acc_watermark_black = self.encrypted_re_embedding(epoch_retrain)

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

    def torch_to_tenseal(self):

        # self.model_encrypted.target_w = self.model.fc2.weight.data.detach().cpu().tolist()
        # self.model_encrypted.target_b = self.model.fc2.bias.data.detach().cpu().tolist()

        self.model_encrypted.target_w = self.model.classifier[-1][2].weight.data.detach().cpu().tolist()
        self.model_encrypted.target_b = self.model.classifier[-1][2].bias.data.detach().cpu().tolist()

        self.model_encrypted.dA_w = self.detector.fc1.weight.data.detach().cpu().tolist()
        self.model_encrypted.dA_b = self.detector.fc1.bias.data.detach().cpu().tolist()

        self.model_encrypted.dB_w = self.detector.fc2.weight.data.detach().cpu().tolist()
        self.model_encrypted.dB_b = self.detector.fc2.bias.data.detach().cpu().tolist()

    def tenseal_to_torch(self):
        # self.model.fc2.weight = torch.nn.parameter.Parameter(
        #     torch.tensor(self.model_encrypted.target_w, device=DEVICE))
        # self.model.fc2.bias = torch.nn.parameter.Parameter(
        #     torch.tensor(self.model_encrypted.target_b, device=DEVICE))

        self.model.classifier[-1][2].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.target_w, device=DEVICE))
        self.model.classifier[-1][2].bias = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.target_b, device=DEVICE))

        self.detector.fc1.weight = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.dA_w, device=DEVICE))
        self.detector.fc1.bias = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.dA_b, device=DEVICE))
        self.detector.fc2.weight = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.dB_w, device=DEVICE))
        self.detector.fc2.bias = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.dB_b, device=DEVICE))

    def encrypted_pre_embedding(self, max_round):

        criterion = torch.nn.MSELoss(reduction="sum")

        print("\nWatermark Pre-Embedding :")

        self.torch_to_tenseal()

        self.model_encrypted.encrypt(self.ctx_training)

        loop = tqdm(range(max_round))

        accuracy = 0

        first_accuracy = 0

        loss = 0

        for epoch in loop:

            self.trigger_set.shuffle()

            accuracy = 0

            loss = 0

            for i in range(len(self.trigger_set)):
                _, _, data_encrypted, label_encrypted = self.trigger_set[i]
                y_pred = self.model_encrypted.forward_watermarking(data_encrypted)
                y_pred = self.model_encrypted.refresh(y_pred)

                self.model_encrypted.backward(data_encrypted, y_pred, label_encrypted)

                y_pred = self.model_encrypted.to_tensor(y_pred).reshape(-1, 1)
                label_encrypted = self.model_encrypted.to_tensor(label_encrypted).reshape(-1, 1)

                # print(y_pred.flatten(), label_encrypted.argmax().item())

                accuracy += int(y_pred.argmax().item() == label_encrypted.argmax().item())

                loss += criterion(y_pred, label_encrypted)

                self.model_encrypted.decrypt()
                self.model_encrypted.encrypt(self.ctx_training)

                self.model_encrypted.update_parameters_regul()

            loop.set_postfix(
                {
                    "Accuracy": accuracy / len(self.trigger_set),
                    "Loss": round(loss.item() / len(self.trigger_set), 2),
                }
            )

            if epoch == 0:
                first_accuracy = accuracy / len(self.trigger_set)

        self.model_encrypted.decrypt()

        self.tenseal_to_torch()

        return first_accuracy

    def encrypted_re_embedding(self, max_round):

        criterion = torch.nn.MSELoss(reduction="sum")

        print("\nWatermark Re-Embedding :")

        self.torch_to_tenseal()

        self.model_encrypted.encrypt(self.ctx_training)

        loop = tqdm(range(max_round))

        accuracy = 0

        first_accuracy = 0

        loss = 0

        for epoch in loop:

            self.trigger_set.shuffle()

            accuracy = 0

            loss = 0

            for i in range(len(self.trigger_set)):
                _, _, data_encrypted, label_encrypted = self.trigger_set[i]
                y_pred = self.model_encrypted.forward_watermarking(data_encrypted)
                y_pred = self.model_encrypted.refresh(y_pred)

                self.model_encrypted.backward(data_encrypted, y_pred, label_encrypted)

                y_pred = self.model_encrypted.to_tensor(y_pred).reshape(-1, 1)
                label_encrypted = self.model_encrypted.to_tensor(label_encrypted).reshape(-1, 1)

                # print(y_pred.flatten(), label_encrypted.argmax().item())

                accuracy += int(y_pred.argmax().item() == label_encrypted.argmax().item())

                loss += criterion(y_pred, label_encrypted)

                self.model_encrypted.decrypt()
                self.model_encrypted.encrypt(self.ctx_training)

                self.model_encrypted.update_parameters()

            loop.set_postfix(
                {
                    "Accuracy": accuracy / len(self.trigger_set),
                    "Loss": round(loss.item() / len(self.trigger_set), 2),
                }
            )

            if epoch == 0:
                first_accuracy = accuracy / len(self.trigger_set)

        self.model_encrypted.decrypt()

        self.tenseal_to_torch()

        return first_accuracy
