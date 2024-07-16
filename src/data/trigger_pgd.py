import glob

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.setting import TRANSFORM_TEST


class PGDSet(Dataset):
    """
    A custom Dataset class for loading images from the PGDSet directory.

    Attributes:
        images_path (str): The path to the directory containing the images.
        data (list): A list of tuples, each containing the path to an image and its class name.
        img_dim (tuple): The dimensions of the images.
        transform (callable): The transformation to apply to the images.
    """

    def __init__(self):
        """
        Initializes the PGDSet with the path to the images and an empty list for the data.
        It then loads the images from the directory and stores their paths and class names in the data list.
        """
        self.images_path = (
            "./trigger_set/PGDAttack/"
        )
        folder_list = glob.glob(self.images_path + "*")
        self.data = []

        for class_path in folder_list:
            class_name = class_path.split("/")[-1]
            for e, img_path in enumerate(
                    glob.glob(self.images_path + class_name + "/*.jpg")
            ):
                if e >= 10:
                    break
                self.data.append([img_path, class_name])

        print("Size of the trigger set :", len(self.data))

        self.img_dim = (32, 32)
        self.transform = TRANSFORM_TEST

    def __len__(self):
        """
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Returns the image and its class name at the specified index.

        Args:
            item (int): The index of the image.

        Returns:
            tuple: A tuple containing the transformed image and its class name.
        """
        img_path, class_name = self.data[item]
        img = Image.open(img_path)
        class_id = torch.tensor(int(class_name))
        img_tensor = self.transform(img)
        return img_tensor.float(), class_id
