from itertools import product

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from . import AIModel


class LeNet(AIModel):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def test_optimizer():
    model = LeNet()
    # model.train_model('Adam', 0.001, 64)
    # model.evaluate_model('Adam', 0.001, 64) # {'loss': [1.6726406e-06], 'acc': 0.9855}

    # model.train_model('Adadelta', 0.001, 64)
    # model.evaluate_model('Adadelta', 0.001, 64) # {'loss': [1.3049467], 'acc': 0.51515}
    #
    # model.train_model('Lamb', 0.001, 64)
    # model.evaluate_model('Lamb', 0.001, 64) # {'loss': [1.176265], 'acc': 0.5243}
    #
    # model.train_model('SGD', 0.001, 64)
    # model.evaluate_model('SGD', 0.001, 64) # {'loss': [1.0667422], 'acc': 0.5411}
    #
    # model.train_model('Momentum', 0.001, 64)
    # model.evaluate_model('Momentum', 0.001, 64) # {'loss': [0.013127858], 'acc': 0.95475}

    model.train_model('Adam', 0.005, 64)
    model.evaluate_model('Adam', 0.005, 64) # {'loss': [1.6726406e-06], 'acc': 0.9855}

    model.train_model('Adadelta', 0.005, 64)
    model.evaluate_model('Adadelta', 0.005, 64) # {'loss': [1.3049467], 'acc': 0.51515}

    model.train_model('Lamb', 0.005, 64)
    model.evaluate_model('Lamb', 0.005, 64) # {'loss': [1.176265], 'acc': 0.5243}

    model.train_model('SGD', 0.005, 64)
    model.evaluate_model('SGD', 0.005, 64) # {'loss': [1.0667422], 'acc': 0.5411}

    model.train_model('Momentum', 0.005, 64)
    model.evaluate_model('Momentum', 0.005, 64) # {'loss': [0.013127858], 'acc': 0.95475}


if __name__ == '__main__':
    # test_optimizer()

    for learning_rate, batch_size in list(product(np.arange(0.001, 0.01, 0.001), [16, 32, 64, 128, 256, 512])):
        model = LeNet()
        model.train_model('Momentum', learning_rate, batch_size)
        model.evaluate_model('Momentum', learning_rate, batch_size)

    # model = LeNet()
    # model.train_model('Adam', 0.01, 64)
    # model.evaluate_model('Adam', 0.01, 64) # {'loss': [2.3033571], 'acc': 0.1135}

    # model.train_model('Adam', 0.001, 64)
    # model.evaluate_model('Adam', 0.001, 64) # {'loss': [1.6726406e-06], 'acc': 0.9855}

    # model.train_model('Adam', 0.001, 128)
    # model.evaluate_model('Adam', 0.001, 128) # {'loss': [1.0132745e-06], 'acc': 0.9824}

    # model.train_model('Adam', 0.006, 128)
    # model.evaluate_model('Adam', 0.006, 128) # {'loss': [0.08492789], 'acc': 0.9404}


