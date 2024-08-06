import numpy as np
import tenseal as ts
from sklearn.datasets import make_classification
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, ctx_training, num_classses=2, num_samples=100, num_features=10):
        self.num_samples = num_samples
        self.data = np.random.rand(num_samples, num_features)
        self.labels = np.random.randint(0, num_classses, num_samples)

        self.data_encrypted = [ctx_training.encrypt(x) for x in self.data]
        self.labels_encrypted = [ctx_training.encrypt(x) for x in self.labels]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.labels[idx],
            self.data_encrypted[idx],
            self.labels_encrypted[idx],
        )


class RandomTriggerSet(Dataset):
    def __init__(self, ctx_training, num_samples=100, num_features=10, num_classes=2):
        self.num_samples = num_samples
        # self.data = np.random.rand(num_samples, num_features)
        # self.data = (self.data-np.mean(self.data))/np.std(self.data)
        # self.labels = np.eye(num_classes)[np.random.randint(0, num_classes, num_samples)]
        # self.labels = np.random.randint(0, 2, num_samples) * 2 - 1

        self.data, self.labels = make_classification(
            n_features=num_features,
            n_redundant=0,
            n_informative=3,
            n_clusters_per_class=1,
            n_classes=num_classes,
            n_samples=num_samples,
        )

        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        self.labels = np.eye(num_classes)[self.labels]

        self.data_encrypted = [
            ts.ckks_tensor(ctx_training, x.reshape(-1, 1).tolist()) for x in self.data
        ]
        self.labels_encrypted = [
            ts.ckks_tensor(ctx_training, x.reshape(-1, 1).tolist()) for x in self.labels
        ]

        self.num_samples = num_samples

    def shuffle(self):
        idx = np.random.permutation(self.num_samples)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.data_encrypted = [self.data_encrypted[i] for i in idx]
        self.labels_encrypted = [self.labels_encrypted[i] for i in idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.labels[idx],
            self.data_encrypted[idx],
            self.labels_encrypted[idx],
        )
