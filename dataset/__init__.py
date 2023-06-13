import os

import PIL
import paddle
from paddle.io import Dataset
from tqdm import tqdm


class MyDataset(Dataset):

    def __init__(self, mode='train', transform=None):
        super(MyDataset, self).__init__()
        self.images = []
        self.labels = []

        for image, label in tqdm(paddle.vision.datasets.MNIST(mode=mode), desc='Initializing dataset'):
            revert_image = PIL.ImageOps.invert(image)
            if transform is not None:
                image = transform(image)
                revert_image = transform(revert_image)
            self.images.append(image)
            self.labels.append(label)
            self.images.append(revert_image)
            self.labels.append(label)

        self.transform = transform

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)