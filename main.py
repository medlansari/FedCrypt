from time import time

from src.federated_learning.server_real_fhe import Server_Real_FHE
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.setting import MAX_ROUNDS, NB_CLIENTS

def main():
    time_save = str(time())
    server = Server_Simulated_FHE("VGG", "CIFAR10", NB_CLIENTS, id=time_save)
    server.train(MAX_ROUNDS, 1e-2, 1e-3,1e-3)
    # server = Server_Real_FHE("VGG", "CIFAR10", NB_CLIENTS, id=time_save)
    # server.train(MAX_ROUNDS, 1e-2, 5, 3)

if __name__ == "__main__":
    main()