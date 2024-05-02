import unittest
import tenseal as ts
import torch

from src.model.dnn import DNN, Detector
from src.model.encyrpted_dnn import EncryptedDNN


class MyTestCase(unittest.TestCase):
    def test_something(self):
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        # create TenSEALContext
        ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        ctx_training.global_scale = 2 ** 21
        ctx_training.generate_galois_keys()

        model_real = DNN(784, False)
        detector_real = Detector()

        model_enc = EncryptedDNN(model_real, detector_real, 10)

        x = ts.ckks_tensor(ctx_training, torch.ones(1,784).tolist())

        model_enc.encrypt(ctx_training)

        y = model_enc.forward_watermarking(x)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
