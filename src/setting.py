import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of workers for the dataloader
MAX_EPOCH_CLIENT = 5  # Number of epochs for each client
NB_CLIENTS = 10  # Number of clients
PRCT_TO_SELECT = 0.5  # Percentage of clients selected for each round
LEARNING_RATE_CLIENT = 1e-2  # Learning rate for the client
BATCH_SIZE_CLIENT = 32  # Batch size for the client
BATCH_SIZE_TRIGGER = 2  # Batch size for the trigger
BATCH_SIZE_SERVER = 20  # Batch size for the server
MAX_ROUNDS = 100  # Number of rounds
TYPE = torch.float32  # Type of the tensor

TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.25, contrast=0.8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

TRANSFORM_TRAIN_MNIST = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

TRANSFORM_TEST_MNIST = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

TRANSFORM_TRAIN_MNIST2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

TRANSFORM_TEST_MNIST2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
