import numpy as np
import torch
import torchvision
from torch.utils.data import random_split, Subset

from src.data.dataset_covid import Covid_dataset
from src.data.non_iid import distribute_non_iid_data
from src.setting import (
    NUM_WORKERS,
    BATCH_SIZE_CLIENT,
    BATCH_SIZE_SERVER,
    TRANSFORM_TRAIN,
    TRANSFORM_TEST,
    TRANSFORM_TRAIN_MNIST,
    TRANSFORM_TEST_MNIST,
    TRANSFORM_TEST_MNIST2,
    TRANSFORM_TRAIN_MNIST2, TRANSFORM_TRAIN_COVID, TRANSFORM_TEST_COVID,
)


def data_splitter(
    dataset: str, nb_clients: int, distrib: str = "IID"
):
    """
    Splits the specified dataset into subsets for each client.

    Args:
        distrib:
        dataset (str): The name of the dataset to split. Currently supports "CIFAR10", "MNIST", and "MNIST2".
        nb_clients (int): The number of clients to split the dataset for.

    Returns:
        tuple: A tuple containing the following elements:
            - subsets_loader (list): A list of DataLoader instances, each containing a subset of the training data for a client.
            - subset_size (np.array): An array containing the size of each subset.
            - test_loader (DataLoader): A DataLoader instance for the test data.

    Raises:
        ValueError: If the specified dataset is not supported.
    """

    batch_size = BATCH_SIZE_CLIENT
    print("Selected Dataset : ", dataset, "\n")

    match dataset:
        case "CIFAR10":

            train_set = torchvision.datasets.CIFAR10(
                root="~/data/", train=True, download=True, transform=TRANSFORM_TRAIN
            )

            test_set = torchvision.datasets.CIFAR10(
                root="~/data/", train=False, download=True, transform=TRANSFORM_TEST
            )
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            num_classes = 10

        case "CIFAR100":

            train_set = torchvision.datasets.CIFAR100(
                root="~/data/", train=True, download=True, transform=TRANSFORM_TRAIN
            )

            test_set = torchvision.datasets.CIFAR100(
                root="~/data/", train=False, download=True, transform=TRANSFORM_TEST
            )
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            num_classes = 100

        case "MNIST":

            train_set = torchvision.datasets.MNIST(
                root="~/data/", train=True, download=True, transform=TRANSFORM_TRAIN_MNIST
            )

            test_set = torchvision.datasets.MNIST(
                root="~/data/", train=False, download=True, transform=TRANSFORM_TEST_MNIST
            )
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            num_classes = 10

        case "FMNIST":

            train_set = torchvision.datasets.FashionMNIST(
                root="~/data/", train=True, download=True, transform=TRANSFORM_TRAIN_MNIST
            )

            test_set = torchvision.datasets.FashionMNIST(
                root="~/data/", train=False, download=True, transform=TRANSFORM_TEST_MNIST
            )
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            num_classes = 10

        case "MNIST2":

            train_set = torchvision.datasets.MNIST(
                root="~/data/", train=True, download=True, transform=TRANSFORM_TRAIN_MNIST2
            )

            indices = torch.concat(
                [
                    torch.where(train_set.targets == 3)[0],
                    torch.where(train_set.targets == 8)[0],
                ],
                dim=0,
            )

            train_set.data, train_set.targets = (
                train_set.data[indices],
                train_set.targets[indices],
            )

            train_set.data = train_set.data.reshape(-1, 784)

            train_set.targets = torch.where(
                train_set.targets == train_set.targets[0].item(), 0, 1
            )

            test_set = torchvision.datasets.MNIST(
                root="~/data/", train=False, download=True, transform=TRANSFORM_TEST_MNIST2
            )

            indices = torch.concat(
                [
                    torch.where(test_set.targets == 3)[0],
                    torch.where(test_set.targets == 8)[0],
                ],
                dim=0,
            )

            test_set.data, test_set.targets = (
                test_set.data[indices],
                test_set.targets[indices],
            )

            test_set.targets = torch.where(test_set.targets == 3, 0, 1)

            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

        case "COVID":

            dataset = Covid_dataset("~/data/covid")

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_set, test_set = random_split(dataset,[train_size, test_size])

            train_set.dataset.transform = TRANSFORM_TRAIN_COVID
            test_set.dataset.transform = TRANSFORM_TEST_COVID

            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE_SERVER,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            num_classes = 3

        case _ :
            raise ValueError(f"Dataset '{dataset}' not found.")


    distrib = "NON-IID"

    print("NON IID ######################################################################################")

    match distrib:

        case "IID":

            subsets_loader = []

            subsets_size = int(len(train_set) / nb_clients)

            if len(train_set) % nb_clients:
                extra = len(train_set) % nb_clients
                # train_set.data, train_set.targets = (
                #     train_set.data[:-extra],
                #     train_set.targets[:-extra],
                # )
                indices = list(range(len(train_set) - extra))
                train_set = Subset(train_set, indices)

            subset_size = [subsets_size for i in range(nb_clients)]

            generator1 = torch.Generator().manual_seed(42)

            for i, subset_loader in enumerate(
                torch.utils.data.random_split(
                    train_set,
                    [subsets_size for _ in range(nb_clients)],
                    generator=generator1,
                )
            ):
                subsets_loader.append(
                    torch.utils.data.DataLoader(
                        subset_loader,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                    )
                )

        case "NON-IID":

            _, subsets_size  = distribute_non_iid_data(train_set, nb_clients, 1, TRANSFORM_TRAIN)

            indices = list(range(sum(subsets_size)))
            train_set = Subset(train_set, indices)

            generator1 = torch.Generator().manual_seed(42)

            subsets_loader = []

            for i, subset_loader in enumerate(
                    torch.utils.data.random_split(
                        train_set,
                        subsets_size,
                        generator=generator1,
                    )
            ):
                subsets_loader.append(
                    torch.utils.data.DataLoader(
                        subset_loader,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                    )
                )

        case _:

            raise ValueError(f"Distribution '{distrib}' not found.")

    print("Size of the train set for each client :", subsets_size)

    print("Size of the test set :", len(test_set), "\n")

    return subsets_loader, np.array(subsets_size), test_loader, num_classes
