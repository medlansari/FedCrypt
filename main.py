from time import time
import argparse
import yaml

from src.federated_learning.server_real_fhe import Server_Real_FHE
from src.federated_learning.server_simulated_fhe import Server_Simulated_FHE

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encrypted', action='store_true', help="Use encrypted model")
    parser.add_argument('--plaintext', dest='encrypt', action='store_false',  help="Use plaintext model")
    parser.set_defaults(encrypt=True)
    parser.add_argument("--cfgFl", type=str, default="", help="Config. file path for the FL settings")
    parser.add_argument("--cfgFhe", type=str, default="", required=False, help="Config. file path for the FHE scheme")

    args = parser.parse_args()

    configFl = yaml.safe_load(open(args.cfgFl, 'r'))

    id = str(time())

    if args.encrypted:
        print("----> Dynamic Watermarking using Real FHE <----\n")
        configFhe = yaml.safe_load(open(args.cfgFhe, 'r'))
        server = Server_Real_FHE(configFl["model"], configFl["dataset"], configFl["fl"]["nb_clients"], configFhe, id)
        server.train(configFl["fl"]["max_round"], configFl["fl"]["lr_clients"], configFl["watermarking"]["max_ep_pretrain"], configFl["watermarking"]["max_ep_retrain"])

    else:
        print("----> Dynamic Watermarking using Simulated FHE <----\n")
        server = Server_Simulated_FHE(configFl["model"], configFl["dataset"], configFl["fl"]["nb_clients"], id)
        server.train(configFl["fl"]["max_round"], float(configFl["fl"]["lr_clients"]), tuple(map(float,configFl["watermarking"]["lr_pretrain"])), tuple(map(float,configFl["watermarking"]["lr_retrain"])))


if __name__ == "__main__":
    main()