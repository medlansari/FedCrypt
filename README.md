# FedCrypt

This repository contains the implementation of the paper "FedCrypt: A Dynamic White-Box Watermarking Scheme for Homomorphic Federated Learning".

The paper is under review to the IEEE Transactions on Dependable and Secure Computing journal.

<figure>
  <img
  src="overview.png"
  alt="The beautiful MDN logo.">
  <figcaption>Figure 1 : Overview of the FedCrypt scheme.</figcaption>
</figure>

## Introduction

*FedCrypt* the first dynamic white-box watermarking technique compatible with HE in FL. FedCrypt involves training a 
projection function on the activations of the encrypted model using a trigger set, preserving client privacy and enabling 
new verification protocols for joint ownership proof between the server and clients without disclosing private information. 

## The FedCrypt Algorithm

The *FedCrypt* algorithm consists on embedding a watermark in the encrypted model from the server-side.

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 50%;">
    <p>To do so, the server </p>
  </div>
  <div style="flex: 50%;">
    <figure>
      <img src="watermarking.png" alt="The beautiful MDN logo." style="width: 100%;">
      <figcaption>Figure 2 : Illustration of the difference between the training process and the watermarking process on a 
four-layer neural network.</figcaption>
    </figure>
  </div>
</div>

<div style="display: flex; align-items: center; justify-content: space-between;">
  
  <div style="flex: 50%;">
    <figure>
      <img src="trigger-set_construction.png" alt="The beautiful MDN logo." style="width: 100%;">
      <figcaption>Figure 3 : Illustration of the possible generation and corresponding verification protocols that can be 
performed using FedCrypt.</figcaption>
    </figure>
  </div>
    <div style="flex: 50%;">
    <p>Le texte que vous souhaitez afficher à gauche de l'image.</p>
  </div>
</div>


## Setup

### Requirements

- Python 3.10 or higher
- matplotlib==3.8.4
- numpy==2.0.0
- Pillow==10.4.0
- scikit_learn==1.4.2
- tenseal==0.3.14
- torch==2.3.0
- torchvision==0.18.0
- tqdm==4.66.2

### Installation

Create a virtual environment and run activate it :

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages :

```bash
pip install -r requirements.txt
```

## Training

The FL training can be performed using the following command :

```bash
python main.py
```

Where ```main.py``` contains the configuration about the training process.

The user can choose between the class ```Server_Real_FHE``` and ```Server_Simulated_FHE``` to perform the training
using the real or simulated FHE, respectively.

For each experiment the user can create an instance of the class ```Server_*_FHE``` as follows :

```python
server = Server_*_FHE(model, dataset, nb_clients)
```

Where :

- ```model``` is the model to be trained among those which are listed in the ```./src/model/model_choice.py``` file. 
- ```dataset``` is the dataset to be used among those which are listed in the ```./src/dataset/data_splitter.py``` file.
- ```nb_clients``` is the number of clients to be used in the FL process.

Then the FL training can be performed using the following line :

```python
server.train(max_rounds, lr_client, lr_pretrain, lr_retrain)
```

Where :
- ```max_rounds``` is the number of rounds to be performed in the FL process. 
- ```lr_client``` is the learning rate used by the clients during the training process.
- ```lr_pretrain``` are the learning rates used by the server during the pre-embedding of the watermark.
- ```lr_retrain``` are the learning rates used by the server during the embedding of the watermark.

### Real FHE

Real FHE use the TenSEAL library to perform the watermark embedding using encrypted parameters. The parameters related 
to the CKKS cryptosystem can be found in the constructor of ```Server_Real_FHE``` in
```./src/federated_learning/server_real_fhe.py``` file.

### Simulated FHE

Simulated FHE use only the PyTorch library to perform the watermark embedding using plaintext parameters.

## Removal Attacks

The removal attacks can be performed using the following command :

```bash
python removal_attack.py
```

Where ```removal_attack.py``` contains the configuration about the removal attack process. The user can perform one of the
following attacks :

- Fine-tuning
- Pruning
- Overwriting

## Citation

Mohammed Lansari, Reda Bellafqira, Katarzyna Kapusta, et al. FedCrypt: A Dynamic White-Box Watermarking Scheme for Homomorphic Federated Learning. TechRxiv.
DOI: Pending approval