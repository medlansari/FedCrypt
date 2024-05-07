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
from src.model.encyrpted_dnn import EncryptedDNN
from src.plot import plot_FHE
from src.setting import NUM_WORKERS, DEVICE, PRCT_TO_SELECT, MAX_EPOCH_CLIENT


class Server_Real_FHE:
    def __init__(self, model: str, dataset: str, nb_clients: int, id: str):
        self.dataset = dataset
        self.nb_clients = nb_clients
        self.num_classes = 2

        self.model = DNN(32, self.num_classes, False)
        self.model.to(DEVICE)

        self.train_subsets, self.subset_size, self.test_set = data_splitter(self.dataset, self.nb_clients)
        self.detector = Detector(self.num_classes)
        self.detector.to(DEVICE)
        # self.model_test = ConvNet(False) TODO
        # self.model_test.to(DEVICE) TODO

        self.id = id
        self.max_round = 30

        # parameters
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        # create TenSEALContext
        self.ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        self.ctx_training.global_scale = 2 ** 21
        self.ctx_training.generate_galois_keys()

        self.trigger_set = RandomTriggerSet(self.ctx_training, 20, 32, self.num_classes)
        self.model_encrypted = EncryptedDNN(self.model, Detector(self.num_classes), 2, self.ctx_training)

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)

    def train(self, nb_rounds: int, lr_client: float, max_rounds : int) -> None:
        print("#" * 60 + " Dynamic Watermarking for Encrypted Model " + "#" * 60)

        acc_test_list = []
        acc_watermark_black_list = []

        self.encrypted_pre_embedding(max_rounds)

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

            acc_watermark_black = self.encrypted_re_embedding(self.max_round)

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

    def torch_to_tenseal(self):
        self.model_encrypted.fc1_weight = self.model.fc1.weight.data.detach().cpu().tolist()
        self.model_encrypted.fc1_bias = self.model.fc1.bias.data.detach().cpu().tolist()

        self.model_encrypted.detect_weight = self.detector.fc1.weight.data.detach().cpu().tolist()
        self.model_encrypted.detect_bias = self.detector.fc1.bias.data.detach().cpu().tolist()

    def tenseal_to_torch(self):
        self.model.fc1.weight = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.fc1_weight, device=DEVICE))
        self.model.fc1.bias = torch.nn.parameter.Parameter(torch.tensor(self.model_encrypted.fc1_bias, device=DEVICE))
        self.detector.fc1.weight = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.detect_weight, device=DEVICE))
        self.detector.fc1.bias = torch.nn.parameter.Parameter(
            torch.tensor(self.model_encrypted.detect_bias, device=DEVICE))

    # def plain_train(self):


    def encrypted_pre_embedding(self, max_round):

        criterion = torch.nn.MSELoss(reduction="sum")

        print("\nWatermark Pre-Embedding :")

        self.torch_to_tenseal()

        self.model_encrypted.encrypt(self.ctx_training)

        for epoch in (range(max_round)):

            accuracy = 0

            loss = 0

            loop = tqdm(range(len(self.trigger_set)))

            pred_array = []
            groundtruth_array = []

            for e, i in enumerate(loop):

                # self.trigger_set.shuffle()

                _, _, data_encrypted, label_encrypted = self.trigger_set[i]
                a = time()
                y_pred = self.model_encrypted.forward_watermarking(data_encrypted)
                y_pred = self.model_encrypted.refresh(y_pred)
                self.model_encrypted.backward_fc1(data_encrypted, y_pred, label_encrypted)
                self.model_encrypted.backward_detect(data_encrypted, y_pred, label_encrypted)

                # b = time() - a
                # print("Time for embedding", round(b, 2))

                y_pred = self.model_encrypted.to_tensor(y_pred).reshape(-1,1)
                label_encrypted = self.model_encrypted.to_tensor(label_encrypted).reshape(-1,1)

                # print(round(y_pred.item(),2), round(label_encrypted.item(),2))
                print(y_pred.flatten(), label_encrypted.argmax().item())

                # loop.set_postfix(
                #     {
                #         "Predicted": int(y_pred.item() > 0),
                #         "Groundtruth": int(label_encrypted.item() > 0),
                #     }
                # )

                accuracy += y_pred.argmax().item() == label_encrypted.argmax().item()

                # accuracy += (torch.abs(y_pred - label_encrypted) < 0.5)

                loss += criterion(y_pred, label_encrypted)

                self.model_encrypted.decrypt()
                self.model_encrypted.encrypt(self.ctx_training)


                self.model_encrypted.update_parameters()


            print(f"\nLoss at epoch {epoch} :", round(loss.item()/len(self.trigger_set),2))
            print(accuracy/len(self.trigger_set))

            print(pred_array)
            print(groundtruth_array)

        self.model_encrypted.decrypt()

        self.tenseal_to_torch()

        print("\n" + 60 * "#" + "\n")

    def encrypted_re_embedding(self, max_round):

        print("\nWatermark Embedding :")

        self.torch_to_tenseal()

        self.model_encrypted.encrypt(self.ctx_training)
        for i in range(max_round):
            a = time()
            self.model_encrypted.backward_fc1()
            self.model_encrypted.backward_detect()
            self.model_encrypted.update_parameters()
            b = time() - a
            print("Time for embedding", round(b, 2))

        self.model_encrypted.decrypt()

        self.tenseal_to_torch()

        print("\n" + 60 * "#" + "\n")

        return 0
