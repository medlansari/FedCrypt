import os
import cv2

from torch.utils.data import Dataset


class Covid_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.classes = ['COVID', 'Non_COVID', 'Normal']
        self.data = []
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}
        for clazz in self.classes:
            class_dir = os.path.join(self.root_dir, clazz)
            class_id = self.class_to_id[clazz]
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                self.data.append((img_path, class_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        img_path, label = self.data[id]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return None, None

        if self.transform:
            transform = self.transform(image=img)
            img = transform['image']

        return img, label