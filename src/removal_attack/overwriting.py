import torch

from src.data.trigger_pgd import PGDSet
from src.data.trigger_wafflepattern import WafflePattern
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.model.convmixer import Detector
from src.setting import NUM_WORKERS

path = "./outputs"


def overwriting_attack(model_name, input_size, num_classes_task, max_epoch, lr, id):
    original_watermark_set = torch.utils.data.DataLoader(
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

    for i in range(5):
        original_detector = Detector(num_classes_task)
        original_detector.load_state_dict(torch.load(path + "/detector_" + id + ".pth"))
        original_detector.to("cuda")

        Server = Server_Simulated_FHE(model_name, "CIFAR10", 10, id)

        Server.trigger_set = new_watermark_set

        Server.model.load_state_dict(torch.load(path + "/save_" + id + ".pth"))
        Server.model_linear.load_state_dict(torch.load(path + "/save_" + id + ".pth"))

        Server.train_overwriting(
            original_watermark_set,
            original_detector,
            max_epoch,
            1e-3,
            (1e-2, 1e-1),
            (1e-2, 1e-1),
        )
