import sys
from itertools import product

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tqdm import tqdm

from dataset import train_dataset, test_dataset

import paddle

__all__ = []

from paddle.nn import Conv2D, MaxPool2D, BatchNorm2D, Linear



class VGG(paddle.nn.Layer):
    def __init__(self):
        super(VGG, self).__init__()

        in_channels = [1, 64, 128, 256, 512, 512]

        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)     #out 224
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)     #  224

        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)

        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)

        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)

        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())

        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, 10)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        # x = self.pool(x)    #112

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        # x = self.pool(x)    #56

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        # x = self.pool(x)    #28

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


from utils import redirect_stdout

EPOCHS = 20
EPOCHS = 1
# build model

def train(batch_size, learning_rate):
    # with open(f'AlexNet-Adam-{learning_rate}-{batch_size}.txt', 'w') as f:
    #     with redirect_stdout(f):
            # print(f'{batch_size}-{learning_rate}')
            # network = AlexNet()
            network = VGG()
            model = paddle.Model(network)

            def train_model():
                model.prepare(paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()),
                              paddle.nn.CrossEntropyLoss(),
                              paddle.metric.Accuracy())
                # 模型训练
                model.fit(train_dataset, epochs=EPOCHS, batch_size=batch_size, verbose=1)
                # 保存模型
                model.save(f'./output/{learning_rate}/{batch_size}/VGG-Adam')

            train_model()
            # 加载模型
            # model.load(f'./output/{learning_rate}/{batch_size}/VGG-Adam')

            # 模型评估
            model.prepare(paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()),
                          paddle.nn.CrossEntropyLoss(),
                          paddle.metric.Accuracy())
            model.evaluate(test_dataset, batch_size=batch_size, verbose=1)


if __name__ == '__main__':
    train(128, 0.01)
    # for batch_size, learning_rate in tqdm(list(product(range(1, 60000, 10), np.arange(0.1, 1.0, 0.1)))):
    #     train(batch_size, learning_rate)
