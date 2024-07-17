from copy import deepcopy

import numpy as np
import tenseal as ts
import torch


class EncryptedModel():

    def __init__(self, torch_dnn, torch_detector, number_class, context_training):

        torch_dnn = torch_dnn.classifier[-1][2]

        self.target_w_shape = torch_dnn.weight.data.shape
        self.target_w = torch_dnn.weight.data.tolist()
        self.target_b = torch_dnn.bias.data.tolist()

        self.target_w_org = torch_dnn.weight.data.tolist()
        self.target_b_org = torch_dnn.bias.data.tolist()

        # self.target_w_shape = torch_dnn.fc2.weight.data.shape
        # self.target_w = torch_dnn.fc2.weight.data.tolist()
        # self.target_b = torch_dnn.fc2.bias.data.tolist()

        self.dA_w_shape = torch_detector.fc1.weight.data.shape
        self.dA_w = torch_detector.fc1.weight.data.tolist()
        self.dA_b = torch_detector.fc1.bias.data.tolist()

        self.dB_w_shape = torch_detector.fc2.weight.data.shape
        self.dB_w = torch_detector.fc2.weight.data.tolist()
        self.dB_b = torch_detector.fc2.bias.data.tolist()

        self.number_class = number_class

        self.z1 = 0
        self.z2 = 0
        self.z3 = 0

        self._delta_target_w = 0
        self._delta_target_b = 0
        self._delta_dA_w = 0
        self._delta_dA_b = 0
        self._delta_dB_w = 0
        self._delta_dB_b = 0
        self._count = 0

        self.ctx = context_training

    def forward_watermarking(self, x):
        z1 = self.target_w.mm(x) + self.target_b
        self.z1 = deepcopy(z1)
        z2 = self.dA_w.mm(z1) + self.dA_b
        self.z2 = deepcopy(z2)
        a = self.relu(z2)
        self.a = deepcopy(a)
        z3 = self.dB_w.mm(a) + self.dB_b
        return z3

    def backward(self, x, y_pred, y_ground):
        diff = y_pred - y_ground

        self._delta_dB_w = (2 / self.number_class) * diff.mm(self.z2.transpose())
        self._delta_dB_b = (2 / self.number_class) * diff

        part1 = self.dB_w.transpose().mm(diff)
        part2 = part1.mul(self.relu_derivated(self.z2))

        self._delta_dA_w = (2 / self.number_class) * part2.mm(self.z1.transpose())
        self._delta_dA_b = (2 / self.number_class) * part2

        part3 = self.dA_w.transpose().mm(part2)

        part3 = self.refresh(part3)

        self._delta_target_w = (2 / self.number_class) * part3.mm(x.transpose())
        self._delta_target_b = (2 / self.number_class) * part3

        self._count += 1

    def update_parameters_regul(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")

        # self._delta_target_w = self.refresh(self._delta_target_w)
        # self._delta_target_b = self.refresh(self._delta_target_b)
        # self._delta_dA_w = self.refresh(self._delta_dA_w)
        # self._delta_dA_b = self.refresh(self._delta_dA_b)
        # self._delta_dB_w = self.refresh(self._delta_dB_w)
        # self._delta_dB_b = self.refresh(self._delta_dB_b)

        lr = 1e-3

        self.target_w -= lr * self._delta_target_w + (self.target_w_org - self.target_w) * 0.05
        self.target_b -= lr * self._delta_target_b.reshape([self.target_w_shape[0]]) + (
                self.target_b_org - self.target_b) * 0.05
        self._delta_fc1_w = 0
        self._delta_fc1_b = 0

        lr = 1e-2

        self.dA_w -= lr * self._delta_dA_w
        self.dA_b -= lr * self._delta_dA_b.reshape([self.dA_w_shape[0]])
        self._delta_dA_w = 0
        self._delta_dA_b = 0

        self.dB_w -= lr * self._delta_dB_w
        self.dB_b -= lr * self._delta_dB_b.reshape([self.dB_w_shape[0]])
        self._delta_dB_w = 0
        self._delta_dB_b = 0

        self._count = 0
        self.refresh_all_parameters()

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")

        self._delta_target_w = self.refresh(self._delta_target_w)
        self._delta_target_b = self.refresh(self._delta_target_b)
        self._delta_dA_w = self.refresh(self._delta_dA_w)
        self._delta_dA_b = self.refresh(self._delta_dA_b)
        self._delta_dB_w = self.refresh(self._delta_dB_w)
        self._delta_dB_b = self.refresh(self._delta_dB_b)

        lr = 1e-3

        self.target_w -= lr * (self._delta_target_w + self.target_w * 0.1)
        self.target_b -= lr * (self._delta_target_b.reshape([self.target_w_shape[0]]) + self.target_b * 0.1)
        self._delta_fc1_w = 0
        self._delta_fc1_b = 0

        lr = 1e-2

        self.dA_w -= lr * self._delta_dA_w
        self.dA_b -= lr * self._delta_dA_b.reshape([self.dA_w_shape[0]])
        self._delta_dA_w = 0
        self._delta_dA_b = 0

        self.dB_w -= lr * self._delta_dB_w
        self.dB_b -= lr * self._delta_dB_b.reshape([self.dB_w_shape[0]])
        self._delta_dB_w = 0
        self._delta_dB_b = 0

        self._count = 0
        self.refresh_all_parameters()

    @staticmethod
    def relu(enc_x):
        return enc_x.polyval([0.47, 0.5, 0.09])

    @staticmethod
    def relu_derivated(enc_x):
        return enc_x.polyval([0.5, 2 * 0.09])

    def encrypt(self, context):
        self.target_w = ts.ckks_tensor(context, self.target_w)
        self.target_b = ts.ckks_tensor(context, self.target_b)
        self.dA_w = ts.ckks_tensor(context, self.dA_w)
        self.dA_b = ts.ckks_tensor(context, self.dA_b)
        self.dB_w = ts.ckks_tensor(context, self.dB_w)
        self.dB_b = ts.ckks_tensor(context, self.dB_b)

    def decrypt(self):
        self.target_w = self.target_w.decrypt().tolist()
        self.target_b = self.target_b.decrypt().tolist()
        self.dA_w = self.dA_w.decrypt().tolist()
        self.dA_b = self.dA_b.decrypt().tolist()
        self.dB_w = self.dB_w.decrypt().tolist()
        self.dB_b = self.dB_b.decrypt().tolist()

    def to_numpy(self, x):
        return np.array(x.decrypt().tolist())

    def to_tensor(self, x):
        return torch.tensor(x.decrypt().tolist())

    def refresh(self, x):
        return ts.ckks_tensor(self.ctx, x.decrypt().tolist())

    def refresh_all_parameters(self):
        self.target_w = self.refresh(self.target_w)
        self.target_b = self.refresh(self.target_b)
        self.dA_w = self.refresh(self.dA_w)
        self.dA_b = self.refresh(self.dA_b)
        self.dB_w = self.refresh(self.dB_w)
        self.dB_b = self.refresh(self.dB_b)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
