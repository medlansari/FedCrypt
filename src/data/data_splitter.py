import numpy as np
import torch
import torchvision

from src.setting import (
    NUM_WORKERS,
    BATCH_SIZE_CLIENT,
    BATCH_SIZE_SERVER,
    TRANSFORM_TRAIN,
    TRANSFORM_TEST,
    TRANSFORM_TRAIN_MNIST,
    TRANSFORM_TEST_MNIST,
    TRANSFORM_TEST_MNIST2,
    TRANSFORM_TRAIN_MNIST2,
)


def data_splitter(
    dataset: str, nb_clients: int
) -> tuple[list[torch.utils.data.DataLoader], np.array, torch.utils.data.DataLoader]:
    """
    Splits the specified dataset into subsets for each client.

    Args:
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

    if dataset == "CIFAR10":

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

    elif dataset == "MNIST":

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

    elif dataset == "MNIST2":

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

    else:
        raise ValueError(f"Dataset '{dataset}' not found.")

    subsets_loader = []

    subsets_size = int(len(train_set) / nb_clients)

    if len(train_set) % nb_clients:
        extra = len(train_set) % nb_clients
        train_set.data, train_set.targets = (
            train_set.data[:-extra],
            train_set.targets[:-extra],
        )

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

    print("Size of the train set for each client :", subsets_size, "\n")

    print("Size of the test set :", len(test_set), "\n")

    return subsets_loader, np.array(subset_size), test_loader
