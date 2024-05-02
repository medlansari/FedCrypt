from time import time

from src.federated_learning.server_simulated_fhe import Server_FHE
from src.setting import MAX_ROUNDS, NB_CLIENTS

def main():
    time_save = str(time())
    server = Server_FHE("ConvNet", "CIFAR10", NB_CLIENTS, id=time_save)
    server.train(MAX_ROUNDS, 1e-1, 1e-2,1e-3)

if __name__ == "__main__":
    main()