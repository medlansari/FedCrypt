

from time import time
import argparse
import yaml

from src.removal_attack.fine_tuning import fine_tuning
from src.removal_attack.overwriting import overwriting
from src.removal_attack.pruning import pruning


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", default="")
    parser.add_argument("--cfgFl", type=str, default="", help="Config. file path for the FL settings")
    parser.add_argument("--id", type=str, default="", help="ID of the run")

    args = parser.parse_args()

    configFl = yaml.safe_load(open(args.cfgFl, 'r'))

    match args.type:

        case "fine-tuning":

            fine_tuning(configFl["method"], configFl["model"], configFl["dataset"], args.id)

        case "pruning":
            pruning(configFl["method"], configFl["model"], configFl["dataset"], args.id)
        #
        case "overwriting":

            overwriting(configFl["method"], configFl["model"], configFl["dataset"], args.id)

        case _:
            print("Unrecognized attack")



if __name__ == "__main__":
    main()