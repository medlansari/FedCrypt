import torch

from src.data.data_splitter import data_splitter
from src.data.trigger_wafflepattern import WafflePattern
from src.setting import NUM_WORKERS, DEVICE


class Server_Real_FHE:
    def __init__(self, model: str, dataset: str, nb_clients: int, id: str):

        self.dataset = dataset
        self.nb_clients = nb_clients
        self.poly_client = False
        self.poly_server = True
        self.model = ConvNet(True)
        self.model.to(DEVICE)
        self.train_subsets, self.subset_size, self.test_set = data_splitter(self.dataset, self.nb_clients)
        self.detector = Detector()
        self.detector.to(DEVICE)
        self.model_test = ConvNet(False)
        self.model_test.to(DEVICE)
        self.trigger_set = torch.utils.data.DataLoader(

        WafflePattern(RGB=True, features=True),
        batch_size = 10,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = True,

        )
        self.id = id
        self.max_round = 30

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)