import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class newNet(paddle.nn.Layer):
    def __init__(self):
        super(newNet, self).__init__()

        # 定义卷积层，输出特征通道 out_channels 设置为 20，卷积核的大小 kernel_size 为 5，卷积步长 stride=1，padding=2
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小 kernel_size 为 2，池化步长为 2
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道 out_channels 设置为 20，卷积核的大小 kernel_size 为 5，卷积步长 stride=1，padding=2
        self.conv2 = paddle.nn.Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小 kernel_size 为 2，池化步长为 2
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是 10
        self.fc = paddle.nn.Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用 Relu，全连接层激活函数使用 softmax
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x