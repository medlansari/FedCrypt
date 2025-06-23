#!/bin/bash


python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml --id VGG_80_5_FHE_1750414359.9677467_VGG_CIFAR10_FedCrypt
python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml --id VGG_80_5_FHE_1750412123.2424004_VGG_CIFAR10_FedIPR
python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml --id VGG_80_5_FHE_1750413300.607872_VGG_CIFAR10_FedTracker
