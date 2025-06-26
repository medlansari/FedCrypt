#!/bin/bash


#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml --id VGG_80_5_FHE_1750414359.9677467_VGG_CIFAR10_FedCrypt
#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml --id VGG_80_5_FHE_1750690352.201126_VGG_CIFAR10_FedIPR
#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml --id VGG_80_5_FHE_1750691381.6489832_VGG_CIFAR10_FedTracker

#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml --id ConvMixer_80_5_FHE_1750419125.5412629_ConvMixer_CIFAR100_FedCrypt
#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_convmixer_fedipr.yaml --id ConvMixer_80_5_FHE_1750416211.825189_ConvMixer_CIFAR100_FedIPR
#python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_convmixer_fedtracker.yaml --id ConvMixer_80_5_FHE_1750417698.2164614_ConvMixer_CIFAR100_FedTracker

python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml --id ResNet_80_5_FHE_1750409542.771338_ResNet_MNIST_FedCrypt
python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_resnet_fedipr.yaml --id ResNet_80_5_FHE_1750406135.0600848_ResNet_MNIST_FedIPR
python3 attacks.py --type overwriting --cfgFl configs/federated_learning/simulated_resnet_fedtracker.yaml --id ResNet_80_5_FHE_1750407864.2315993_ResNet_MNIST_FedTracker
