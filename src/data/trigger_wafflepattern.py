import glob

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.setting import TRANSFORM_TEST, TRANSFORM_TEST_MNIST


class WafflePattern(Dataset):
    """
    A custom Dataset class for loading images from the WafflePattern directory.

    Attributes:
        images_path (str): The path to the directory containing the images.
        data (list): A list of tuples, each containing the path to an image and its class name.
        class_count (dict): A dictionary to count the number of images per class.
        img_dim (tuple): The dimensions of the images.
        transform (callable): The transformation to apply to the images.
        RGB (bool): A flag to indicate if the images are in RGB format.
        features (bool): A flag to indicate if the features are to be extracted.
    """

    def __init__(self, RGB=True, features=False):
        """
        Initializes the WafflePattern with the path to the images and an empty list for the data.
        It then loads the images from the directory and stores their paths and class names in the data list.

        Args:
            RGB (bool, optional): A flag to indicate if the images are in RGB format. Defaults to True.
            features (bool, optional): A flag to indicate if the features are to be extracted. Defaults to False.
        """
        self.images_path = "./trigger_set/WafflePattern/"
        file_list = glob.glob(self.images_path + "*")
        self.data = []
        self.class_count = dict.fromkeys([str(i) for i in range(10)], 0)
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])

        print("Size of the trigger set :", len(self.data))

        if not (RGB):
            self.transform = TRANSFORM_TEST_MNIST
            self.img_dim = (28, 28)
        else:
            self.transform = TRANSFORM_TEST
            self.img_dim = (32, 32)

        self.RGB = RGB
        self.features = features

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
        if not (self.RGB):
            img = img.convert("L")
        class_id = torch.tensor(int(class_name))
        img_tensor = self.transform(img)
        return img_tensor.float(), class_id.float()
