from torch import optim, nn
import torch
import torch.nn.functional as F

from src.metric import accuracy, watermark_detection_rate
from src.model.activation import identity
from src.model.convnet import ConvNet
from src.setting import DEVICE, LEARNING_RATE_CLIENT, MAX_EPOCH_CLIENT


class Client:
    def __init__(self, model, weights, train_set):
        self.model = model
        self.model.load_state_dict(weights)
        self.model.to(DEVICE)
        self.train_set = train_set

    def train(
            self,
            lr=LEARNING_RATE_CLIENT,
            max_epoch=MAX_EPOCH_CLIENT,
            test_loader=None,
            trigger_loader=None,
            detector=None
    ):

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        test_array = []

        watermark_array = []

        if not (test_loader is None):
            acc_test, acc_loss = accuracy(self.model, test_loader)

            test_array.append(acc_test)

        if not (trigger_loader is None):
            self.model.activation = identity
            acc_watermark, loss_watermark = watermark_detection_rate(self.model, detector, trigger_loader)
            self.model.activation = F.relu

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

            if not (test_loader is None) and epoch % 5 == 0:
                acc_test, acc_loss = accuracy(self.model, test_loader)

                test_array.append(acc_test)

            if not (trigger_loader is None) and epoch % 5 == 0:
                self.model.activation = identity
                acc_watermark, loss_watermark = watermark_detection_rate(self.model, detector, trigger_loader)
                self.model.activation = F.relu

                watermark_array.append(acc_watermark)

        return test_array, watermark_array
