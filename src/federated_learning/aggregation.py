import torch
from torch import nn

from src.federated_learning.client import Client


def fedavg(clients: list[Client], server_model: nn.Module,
           subset_size: list[int], selected_clients: list[int]) -> None:
    """
    Performs the aggregation of the weights of the selected clients' models using the FedAvg algorithm.

    Args:
        clients (list[Client]): List of all clients.
        server_model (nn.Module): The server's model.
        subset_size (list[int]): List of subset sizes for each client.
        selected_clients (list[int]): List of indices of the clients selected for aggregation.

    Returns:
        None: The function updates the server model's weights in place.
    """
    subset_size = subset_size[selected_clients]
    clients = clients[selected_clients]
    subset_size_sum = sum(subset_size)

    with torch.no_grad():
        server_dict = server_model.state_dict()

        for name_server in server_dict.keys():
            server_dict[name_server].zero_()

            for idx, client in enumerate(clients):
                if client.model.state_dict()[name_server].dtype is torch.long:
                    weight = (
                                     subset_size[idx] / subset_size_sum
                             ) * client.model.state_dict()[name_server].clone().detach()
                    weight = weight.long()

                else:
                    weight = (
                                     subset_size[idx] / subset_size_sum
                             ) * client.model.state_dict()[name_server].clone().detach()

                server_dict[name_server].add_(weight)
