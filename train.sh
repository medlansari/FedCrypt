#!/bin/bash



for i in {1..3}
do
  echo "Run number $i"
  python3 main.py --plaintext --cfgFl configs/federated_learning/simulated_convmixer_fedcrypt.yaml
done

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