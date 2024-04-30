import torch
import tenseal as ts
from torch import nn

from src.model.activation import identity, relu_poly


class DNN(torch.nn.Module):

    def __init__(self, n_features, poly):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 32)
        self.fc2 = torch.nn.Linear(32, 10)

        self.poly = poly
        if poly:
            self.activation = identity
        else:
            self.activation = torch.nn.ReLU

    def forward(self, x, features):
        x = x.view(x.size(0), -1)
        if features:
            return relu_poly(self.fc1(x))
        z = self.activation(self.fc1(x))
        out = self.fc2(z)
        return out

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 10)

        self.activation = identity

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

class EncryptedDNN():

    def __init__(self, torch_dnn, torch_detector, number_class):

        self.fc1_weight_shape = torch_dnn.fc1.weight.data.shape
        self.fc1_weight = torch_dnn.fc1.weight.data.tolist()
        self.fc1_bias = torch_dnn.fc1.bias.data.tolist()
        self.detect_weight_shape = torch_detector.fc1.weight.data.shape
        self.detect_weight = torch_detector.fc1.weight.data.tolist()
        self.detect_bias = torch_detector.fc1.bias.data.tolist()
        self.number_class = number_class

        self._delta_fc1_w = 0
        self._delta_fc1_b = 0
        self._delta_detect_w = 0
        self._delta_detect_b = 0
        self._count = 0

    def forward_watermarking(self, x):
        z = self.relu(self.fc1_weight.mm(x) + self.fc1_bias)
        return self.detect_weight.mm(z) + self.detect_bias

    def backward_watermarking(self, x, y_pred, y_ground):
        diff = y_ground-y_pred
        part1 = self.detect_weight.transpose().mm(diff)
        part2 = part1.mul(self.relu_derivated(self.fc1_weight.mm(x) + self.fc1_bias))
        final = part2.mul(x)

        self._delta_fc1_w = (-2/self.number_class) * final
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        lr = 1e-4
        self.fc1_weight -= lr * self._delta_fc1_w + 1# TODO
        self.bias -= lr * self._delta_b
        self._delta_fc1_w = 0
        self._delta_b = 0
        self._count = 0

    @staticmethod
    def relu(enc_x):
        return enc_x.polyval([0.47, 0.5, 0.09])

    @staticmethod
    def relu_derivated(enc_x):
        return enc_x.polyval([0.5, 2 * 0.09])

    def encrypt(self, context):
        self.fc1_weight = ts.ckks_tensor(context, self.fc1_weight)
        self.fc1_bias = ts.ckks_tensor(context, self.fc1_bias)
        self.detect_weight = ts.ckks_tensor(context, self.detect_weight)
        self.detect_bias = ts.ckks_tensor(context, self.detect_bias)

    def decrypt(self):
        self.fc1_weight = self.fc1_weight.decrypt().tolist()
        self.fc1_bias = self.fc1_bias.decrypt().tolist()
        self.detect_weight = self.detect_weight.decrypt().tolist()
        self.detect_bias = self.detect_bias.decrypt().tolist()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)