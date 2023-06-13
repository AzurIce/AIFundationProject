import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, Layer
import paddle.nn.functional as F
from tqdm import tqdm
from itertools import product
from dataset import train_dataset, test_dataset

class MyAlexNet(Layer):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=48 * 2, kernel_size=5, stride=1, padding=2) # 1x28x28 -padding-> 1x32x32 -conv-> (48x2)x28x28
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2) # (48x2)x14x14
        self.conv2 = Conv2D(in_channels=48 * 2, out_channels=128 * 2, kernel_size=5, stride=1, padding=2) # (48x2)x14x14 -padding-> (48x2)x18x18 -conv-> (128x2)x14x14
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2) # (128x2)x7x7
        self.conv3 = Conv2D(in_channels=128 * 2, out_channels=192 * 2, kernel_size=3, stride=1, padding=1) # (128x2)x7x7 -padding-> (128x2)x9x9 -conv-> (192x2)x7x7
        self.conv4 = Conv2D(in_channels=192 * 2, out_channels=192 * 2, kernel_size=3, stride=1, padding=1) # (192x2)x7x7 -padding-> (192x2)x9x9 -conv-> (192x2)x7x7
        self.conv5 = Conv2D(in_channels=192 * 2, out_channels=128 * 2, kernel_size=3, stride=1, padding=1) # (192x2)x7x7 -padding-> (192x2)x9x9 -conv-> (128x2)x7x7
        self.max_pool5 = MaxPool2D(kernel_size=3, stride=2) # (128x2)x7x7 -pool-> (128x2)x3x3

        self.fc1 = Linear(in_features=2304, out_features=1024)
        self.drop1 = Dropout(0.5)
        self.fc2 = Linear(in_features=1024, out_features=1024)
        self.drop2 = Dropout(0.5)
        self.fc3 = Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool5(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

from utils import redirect_stdout

EPOCHS = 20
batch_size = 128

if __name__ == '__main__':
    for learning_rate in tqdm(np.arange(0.05, 0, -0.01)):
        with open(f'MyAlexNet-Momentum-{learning_rate}-{batch_size}.txt', 'w') as f:
            with redirect_stdout(f):
                # print(f'{batch_size}-{learning_rate}')
                network = MyAlexNet()
                model = paddle.Model(network)


                def train_model():
                    model.prepare(paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=model.parameters()),
                                  paddle.nn.CrossEntropyLoss(),
                                  paddle.metric.Accuracy())
                    # 模型训练
                    model.fit(train_dataset, epochs=EPOCHS, batch_size=batch_size, verbose=1)
                    # 保存模型
                    model.save(f'./output/{learning_rate}/{batch_size}/MyAlexNet-Momentum')


                train_model()
                # 加载模型
                # model.load(f'./output/{learning_rate}/{batch_size}/MyAlexNet-Momentum')

                # 模型评估
                model.prepare(paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=model.parameters()),
                              paddle.nn.CrossEntropyLoss(),
                              paddle.metric.Accuracy())
                model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
