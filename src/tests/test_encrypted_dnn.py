import unittest

import numpy as np
import tenseal as ts
import torch

from src.model.dnn import DNN, Detector
from src.model.encyrpted_dnn import EncryptedDNN


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        # parameters
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        # create TenSEALContext
        ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        ctx_training.global_scale = 2 ** 21
        ctx_training.generate_galois_keys()

        x = torch.randn(32, 1)
        x_enc = ts.ckks_tensor(ctx_training, x.tolist())

        y_ground = torch.ones(2, 1)
        y_ground_enc = ts.ckks_tensor(ctx_training, y_ground.tolist())

        model_real = DNN(32, True)
        detector_real = Detector()
        model_enc = EncryptedDNN(model_real, detector_real, 2, ctx_training)
        model_enc.encrypt(ctx_training)

        criterion = torch.nn.MSELoss(reduction="mean")

        y_pred = detector_real(model_real(x.T, True))
        y_pred_enc = model_enc.forward_watermarking(x_enc)

        y_pred = y_pred.argmax().item()
        y_pred_enc = np.array(y_pred_enc.decrypt().tolist()).argmax()

        self.assertEqual(y_pred, y_pred_enc)

    def test_backward(self):
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        # create TenSEALContext
        ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        ctx_training.global_scale = 2 ** 21
        ctx_training.generate_galois_keys()

        x = torch.randn(32, 1)
        x_enc = ts.ckks_tensor(ctx_training, x.tolist())

        y_ground = torch.ones(2, 1)
        y_ground_enc = ts.ckks_tensor(ctx_training, y_ground.tolist())

        model_real = DNN(32, True)
        detector_real = Detector()
        model_enc = EncryptedDNN(model_real, detector_real, 2, ctx_training)
        model_enc.encrypt(ctx_training)

        criterion = torch.nn.MSELoss(reduction="mean")

        y_pred = detector_real(model_real(x.T, True))
        y_pred_enc = model_enc.forward_watermarking(x_enc)

        loss = criterion(y_pred, y_ground.T)
        loss.backward(retain_graph=True)

        grad_fc1 = model_real.fc1.weight.grad
        grad_fc1 = grad_fc1 + abs(grad_fc1.min())
        grad_fc1 = grad_fc1 / grad_fc1.max()

        grad_fc1_enc = torch.tensor(model_enc.backward_fc1(x_enc, y_pred_enc, y_ground_enc).decrypt().tolist())
        grad_fc1_enc = grad_fc1_enc + abs(grad_fc1_enc.min())
        grad_fc1_enc = grad_fc1_enc / grad_fc1_enc.max()

        diff = abs(grad_fc1 - grad_fc1_enc).max()

        grad_detect = detector_real.fc1.weight.grad
        grad_detect = grad_detect + abs(grad_detect.min())
        grad_detect = grad_detect / grad_detect.max()

        grad_detect_enc = torch.tensor(model_enc.backward_detect(x_enc, y_pred_enc, y_ground_enc).decrypt().tolist())
        grad_detect_enc = grad_detect_enc + abs(grad_detect_enc.min())
        grad_detect_enc = grad_detect_enc / grad_detect_enc.max()

        diff_detect = abs(grad_detect - grad_detect_enc).max()

        self.assertLess(diff, 0.1)
        self.assertLess(diff_detect, 0.2)


if __name__ == '__main__':
    unittest.main()
