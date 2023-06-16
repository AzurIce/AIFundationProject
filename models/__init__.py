import os

import paddle

import dataset

EPOCHS = 20


class AIModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = paddle.Model(self)

    def create_optimizer(self, opt_name, learning_rate):
        if opt_name == 'Adam':
            return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=self.model.parameters())
        elif opt_name == 'Adadelta':
            return paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=self.model.parameters())
        elif opt_name == 'Lamb':
            return paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=self.model.parameters())
        elif opt_name == 'SGD':
            return paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=self.model.parameters())
        elif opt_name == 'Momentum':
            return paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=self.model.parameters())

    def train_model(self, opt_name, learning_rate, batch_size):
        train_dataset = dataset.MyDataset(mode='train')

        self.model.prepare(
            self.create_optimizer(opt_name, learning_rate),
            paddle.nn.CrossEntropyLoss(),
            paddle.metric.Accuracy()
        )

        os.getcwd()

        # 模型训练
        self.model.fit(train_dataset, epochs=EPOCHS, batch_size=batch_size, verbose=1)
        # 保存模型
        self.model.save(f'{os.getcwd()}/{self.__class__.__name__}-{opt_name}-{learning_rate}-{batch_size}/model')

    def load_model(self, opt_name, learning_rate, batch_size):
        self.model.load(f'{os.getcwd()}/LeNet-{opt_name}-{learning_rate}-{batch_size}/model')

        # 模型评估
        self.model.prepare(
            self.create_optimizer(opt_name, learning_rate),
            paddle.nn.CrossEntropyLoss(),
            paddle.metric.Accuracy()
        )

    def evaluate_model(self, opt_name, learning_rate, batch_size):
        test_dataset = dataset.MyDataset(mode='test')

        self.load_model(opt_name, learning_rate, batch_size)
        eval_res = self.model.evaluate(test_dataset, batch_size)
        with open(f'{os.getcwd()}/{self.__class__.__name__}-{opt_name}-{learning_rate}-{batch_size}/eval_res.txt', 'w') as f:
            f.write(f'{eval_res}')
        print(eval_res)
