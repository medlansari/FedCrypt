#!/bin/bash


python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml --id VGG_80_5_FHE_1750414359.9677467_VGG_CIFAR10_FedCrypt
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml --id VGG_80_5_FHE_1750773538.2006395_VGG_CIFAR10_FedCrypt
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml --id VGG_80_5_FHE_1750778050.2022898_VGG_CIFAR10_FedCrypt

python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml --id VGG_80_5_FHE_1750412123.2424004_VGG_CIFAR10_FedIPR
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml --id VGG_80_5_FHE_1750771339.6818657_VGG_CIFAR10_FedIPR
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml --id VGG_80_5_FHE_1750775797.7250917_VGG_CIFAR10_FedIPR

python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml --id VGG_80_5_FHE_1750413300.607872_VGG_CIFAR10_FedTracker
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml --id VGG_80_5_FHE_1750776979.1078675_VGG_CIFAR10_FedTracker
python3 attacks.py --type pruning --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml --id VGG_80_5_FHE_1750772441.4172165_VGG_CIFAR10_FedTracker
