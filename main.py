from time import time

from src.federated_learning.server_real_fhe import Server_Real_FHE
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE
from src.setting import MAX_ROUNDS, NB_CLIENTS

def main():
    time_save = str(time())
    # lr_pretrain = [(1e-3,1e-2),(1e-4,1e-2),(1e-2,1e-2)]
    # lr_retrain = [(1e-3,1e-2),(1e-4,1e-2),(1e-2,1e-2)]
    # for lr_p in lr_pretrain:
    #     for lr_r in lr_retrain:
    #for i in range(5):

    #    time_save = str(time())
    #    server = Server_Simulated_FHE("ResNet", "CIFAR10", NB_CLIENTS, id=time_save)
    #    server.train(MAX_ROUNDS, 1e-3, (1e-2, 1e-1), (1e-2, 1e-1))

    #    time_save = str(time())
    #    server = Server_Simulated_FHE("ConvMixer", "CIFAR10", NB_CLIENTS, id=time_save)
    #    server.train(MAX_ROUNDS, 1e-2, (1e-2, 1e-1), (1e-2, 1e-1))
    # time_save = str(time())
    # server = Server_Simulated_FHE("VGG", "CIFAR10", NB_CLIENTS, id=time_save)
    # server.train(MAX_ROUNDS, 1e-2, (1e-3,1e-2), (1e-3,1e-2))
    for i in range(5):
        time_save = str(time())
        server = Server_Real_FHE("VGG_encrypted", "CIFAR10", NB_CLIENTS, id=time_save)
        # server.train(MAX_ROUNDS, 1e-2, 5, 3)
        server.train(MAX_ROUNDS, 1e-2, 1, 1)


if __name__ == "__main__":
    main()