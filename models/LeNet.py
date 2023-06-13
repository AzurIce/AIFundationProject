import sys
from itertools import product

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tqdm import tqdm

from dataset import train_dataset, test_dataset


class LeNet(nn.Layer):
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


from utils import redirect_stdout

EPOCHS = 20

if __name__ == '__main__':
    for batch_size, learning_rate in tqdm(list(product(range(1, 60000, 10), np.arange(0.1, 1.0, 0.1)))):
        with open(f'LeNet-Adam-{learning_rate}-{batch_size}.txt', 'w') as f:
            with redirect_stdout(f):
                # print(f'{batch_size}-{learning_rate}')
                network = LeNet()
                model = paddle.Model(network)


                def train_model():
                    model.prepare(paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()),
                                  paddle.nn.CrossEntropyLoss(),
                                  paddle.metric.Accuracy())
                    # 模型训练
                    model.fit(train_dataset, epochs=EPOCHS, batch_size=batch_size, verbose=1)
                    # 保存模型
                    model.save(f'./output/{learning_rate}/{batch_size}/LeNet-Adam')


                train_model()
                # 加载模型
                # model.load('output/LeNet-Adam-CrossEntropyLoss')

                # 模型评估
                model.prepare(paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()),
                              paddle.nn.CrossEntropyLoss(),
                              paddle.metric.Accuracy())
                model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
