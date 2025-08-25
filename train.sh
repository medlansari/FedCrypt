#!/bin/bash

#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml

python3 main.py --encrypted --cfgFl configs/federated_learning/real_vgg.yaml --cfgFhe configs/fhe_scheme/without_refresh.yaml


#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedipr.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedtracker.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_classhidden.yaml
#
#python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml
#python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml
#python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml
#python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_vgg_classhidden.yaml
#
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedipr.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedtracker.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_classhidden.yaml

#python3 attacks.py --type fine-tuning --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml --id ConvMixer_80_5_FHE_1750098822.762537_ConvMixer_CIFAR100_FedCrypt

#for i in {1..3}
#do
#  echo "Run number $i"
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml
#done

#for i in {1..3}
#do
#  echo "Run number $i"
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_classhidden.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_vgg_classhidden.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_classhidden.yaml
#done

#for i in {1..3}
#do
#  echo "Run number $i"
#  python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_vgg_wholeaked.yaml
#  python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml
#  python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedtracker.yaml
#  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedipr.yaml
#  python3 main.py  --plaintext --cfgFl configs/federated_learning/simulated_convmixer_wholeaked.yaml
#done


#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedcrypt.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedtracker.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_fedipr.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_wholeaked.yaml
#python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_resnet_classhidden.yaml


#python3 main.py --method FedCrypt --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedcrypt.yaml
#python3 main.py --method FedTracker --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedtracker.yaml
#python3 main.py --method FedIPR --plaintext --cfgFl configs/federated_learning/simulated_vgg_fedipr.yaml
#python3 main.py --method Wholeaked --plaintext --cfgFl configs/federated_learning/simulated_vgg_wholeaked.yaml
#
#python3 main.py --method FedCrypt --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml
#python3 main.py --method FedTracker --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedtracker.yaml
#python3 main.py --method FedIPR --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedipr.yaml
#python3 main.py --method Wholeaked --plaintext --cfgFl configs/federated_learning/simulated_convmixer_wholeaked.yaml