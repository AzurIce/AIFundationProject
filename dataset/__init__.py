import PIL
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision import Normalize
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
            self.images.append(np.expand_dims(np.array(image).astype('float32'), 0))
            self.labels.append(label[0])
            self.images.append(np.expand_dims(np.array(revert_image).astype('float32'), 0))
            self.labels.append(label[0])

        self.transform = transform

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
# transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 打印数据集样本数
# train_dataset = MyDataset(mode='train', transform=transform)
# test_dataset = MyDataset(mode='test', transform=transform)
# print('train_dataset images: ', len(train_dataset), 'test_dataset images: ', len(test_dataset))
