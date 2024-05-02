import torch

from src.data.trigger_pgd import PGDSet
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.server_simulated_fhe import Server_FHE
from src.model.convnet import Detector
from src.setting import NUM_WORKERS

path = "./outputs"


def overwriting_attack(id):
    watermark_set = torch.utils.data.DataLoader(
        WafflePattern(),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    new_watermark_set = torch.utils.data.DataLoader(
        PGDSet(),
        batch_size=10,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    dynamic_key = Detector()
    dynamic_key.load_state_dict(torch.load(path + "/detector_" + id + ".pth"))
    dynamic_key.to("cuda")

    S = Server_FHE("CIFAR10", 10, "iid", alpha=0.0, id="1234")

    S.train_subsets = S.train_subsets[0:9]
    S.subset_size = S.subset_size[0:9]
    S.nb_clients = 8

    S.watermark_set = new_watermark_set

    S.model.load_state_dict(torch.load(path + "/save_" + id + ".pth"))

    S.detector = Detector()
    S.detector.to("cuda")

    S.train(100, 1e-4, 1e-2, 1e-2, (watermark_set, key, message, dynamic_key))

    client_malicious = Client(torch.load(path + "/save_80_5_FHE" + id + ".pth"), train_set, poly=False)

    client_malicious.model.load_state_dict(S.model.state_dict())

    client_malicious.detector = dynamic_key

    test_tmp, dynamic_tmp, static_tmp = get_accuracies(client_malicious.model, test_set, watermark_set, key, message,
                                                       dynamic_key)
