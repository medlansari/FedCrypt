from torch import optim, nn
import torch
import torch.nn.functional as F

from src.metric import accuracy, watermark_detection_rate, one_hot_encoding
from src.model.convnet import ConvNet
from src.model.dnn import DNN
from src.model.model_choice import model_choice
from src.setting import DEVICE, LEARNING_RATE_CLIENT, MAX_EPOCH_CLIENT


class Client:
    """
    The Client class represents a client in a federated learning system. Each client has its own model and training set.

    Attributes:
        model TODO
        train_set (DataLoader): The training set used by the client.

    Methods:
        train(lr=LEARNING_RATE_CLIENT, max_epoch=MAX_EPOCH_CLIENT, test_loader=None, trigger_loader=None, detector=None):
            Trains the client's model using the specified learning rate and maximum number of epochs.
            Optionally, a test loader, trigger loader, and detector can be provided for testing and watermark detection.
    """

    def __init__(self, model: str, weights: dict, input_size, num_classes, train_set: torch.utils.data.DataLoader):
        self.model, self.model_linear = model_choice(model, input_size, num_classes)
        self.model.load_state_dict(weights)
        self.model_linear.load_state_dict(weights)
        self.model.to(DEVICE)
        self.model_linear.to(DEVICE)
        self.train_set = train_set

    def train(
            self,
            lr: float = LEARNING_RATE_CLIENT,
            max_epoch: int = MAX_EPOCH_CLIENT
    ) -> tuple[list[float], list[float]]:

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        test_array = []

        self.model.train()

        for epoch in range(max_epoch):
            accumulate_loss = 0

            for inputs, outputs in self.train_set:
                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs)

                    loss = criterion(outputs_predicted, outputs)

                loss.backward()

                accumulate_loss += loss.item()

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

        return test_array

    def train_fine_tuning(
            self,
            lr: float = LEARNING_RATE_CLIENT,
            max_epoch: int = MAX_EPOCH_CLIENT,
            test_loader: torch.utils.data.DataLoader = None,
            trigger_loader: torch.utils.data.DataLoader = None,
            detector: nn.Module = None
    ) -> tuple[list[float], list[float]]:

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        test_array = []

        watermark_array = []

        acc_test, acc_loss = accuracy(self.model, test_loader)

        test_array.append(acc_test)

        acc_watermark, loss_watermark = watermark_detection_rate(self.model_linear, detector, trigger_loader)

        print("Initial watermark detection rate: ", acc_watermark, "Initial watermark loss: ", loss_watermark)

        watermark_array.append(acc_watermark)

        self.model.train()

        for epoch in range(max_epoch):
            accumulate_loss = 0

            for inputs, outputs in self.train_set:
                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs)

                    loss = criterion(outputs_predicted, outputs)

                loss.backward()

                accumulate_loss += loss.item()

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            acc_test, acc_loss = accuracy(self.model, test_loader)

            test_array.append(acc_test)

            self.model_linear.load_state_dict(self.model.state_dict())

            acc_watermark, loss_watermark = watermark_detection_rate(self.model_linear, detector, trigger_loader)

            print(f"\rEpoch: {epoch}, Acc : {acc_test}, WDR : {acc_watermark}", end='', flush=True)

            watermark_array.append(acc_watermark)

        return test_array, watermark_array

    def train_overwriting(
            self,
            lr: float = LEARNING_RATE_CLIENT,
            max_epoch: int = MAX_EPOCH_CLIENT,
            test_loader: torch.utils.data.DataLoader = None,
            trigger_loader: torch.utils.data.DataLoader = None,
            detector: nn.Module = None,
            original_trigger_loader: torch.utils.data.DataLoader = None,
            original_detector: nn.Module = None
    ) -> tuple[list[float], list[float]]:

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        optimizer_target = optim.SGD(self.model_linear.classifier[4].parameters(), lr=lr_retrain[0])

        optimizer_detector = optim.SGD(self.detector.parameters(), lr=lr_retrain[1])

        criterion = nn.CrossEntropyLoss()

        criterion_watermarking = nn.MSELoss()

        test_array = []

        new_watermark_array = []

        old_watermark_array = []

        acc_test, acc_loss = accuracy(self.model, test_loader)

        test_array.append(acc_test)

        acc_watermark, loss_watermark = watermark_detection_rate(self.model_linear, detector, trigger_loader)

        print("New watermark detection rate: ", acc_watermark, "New watermark loss: ", loss_watermark)

        new_watermark_array.append(acc_watermark)

        acc_watermark, loss_watermark = watermark_detection_rate(self.model_linear, original_detector,
                                                                 original_trigger_loader)

        print("Old watermark detection rate: ", acc_watermark, "Old watermark loss: ", loss_watermark)

        old_watermark_array.append(acc_watermark)

        self.model.train()

        self.model_linear.trainable()

        for epoch in range(max_epoch):
            accumulate_loss = 0

            for inputs, outputs in self.train_set:
                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs)

                    loss = criterion(outputs_predicted, outputs)

                loss.backward()

                accumulate_loss += loss.item()

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            for inputs, outputs in self.trigger_set:
                optimizer_target.zero_grad(set_to_none=True)
                optimizer_detector.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                outputs = one_hot_encoding(outputs)

                with torch.autocast(device_type="cuda"):
                    features_predicted = self.model_linear(inputs)

                    outputs_predicted = detector(features_predicted)

                    blackbox_loss = criterion_watermarking(outputs_predicted, outputs)

                    regul = (1e-1 * self.model_linear.classifier[4].weight.pow(2).sum())

                    loss = blackbox_loss + regul

                loss.backward()

                optimizer_target.step()
                optimizer_detector.step()

                accumulate_loss += loss.item()

            acc_test, acc_loss = accuracy(self.model, test_loader)

            test_array.append(acc_test)

            self.model_linear.load_state_dict(self.model.state_dict())

            acc_watermark, loss_watermark = watermark_detection_rate(self.model_linear, detector, trigger_loader)

            new_watermark_array.append(acc_watermark)

            acc_old_watermark, loss_old_watermark = watermark_detection_rate(self.model_linear, original_detector,
                                                                             original_trigger_loader)

            new_watermark_array.append(acc_watermark)

            old_watermark_array.append(acc_old_watermark)

            print(f"\rEpoch: {epoch}, Acc : {acc_test}, New WDR : {acc_watermark}, Old WDR : {acc_old_watermark}",
                  end='', flush=True)

        return test_array, new_watermark_array, old_watermark_array
