import os
import re

# import paddle
import numpy as np
import pandas as pd

# from models.LeNet import LeNet
# import paddle
# import paddle.nn.functional as F
# from tqdm import tqdm

# from dataset import train_dataset, test_dataset

# EPOCHS = 20


# def train(learn_rate, batch_size):
#     train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#     model = LeNet()
#     opt = paddle.optimizer.Adam(learning_rate=learn_rate, parameters=model.parameters())
#     loss_fun = F.cross_entropy
#
#     x = []
#     train_lost_mean_list = []
#     acc_mean_list = []
#     loss_mean_list = []
#
#     best_result, best_epoch = 0, 0
#     for epoch in range(EPOCHS):
#         # 训练
#         train_loss_list = []
#         for i, (images, labels) in tqdm(list(enumerate(train_loader)), f'epoch-{epoch} batch'):
#             predicts = model(images)
#             acc = paddle.metric.accuracy(predicts, labels)
#             loss = loss_fun(predicts, labels)
#             train_loss_list.append(loss.numpy())
#             loss.backward()
#             # if i % 100 == 0:
#             #     print("epoch: {}, batch_id: {}, loss is: {}, acc is : {}".format(epoch, i, loss.numpy(), acc.numpy()))
#             opt.step()
#             opt.clear_grad()
#
#         train_loss_mean = np.array(train_loss_list).mean()
#         train_lost_mean_list.append(train_loss_mean)
#
#         # 评估
#         acc_list = []
#         loss_list = []
#         for i, (images, labels) in enumerate(test_loader):
#             predicts = model(images)
#
#             acc = paddle.metric.accuracy(predicts, labels)
#             loss = loss_fun(predicts, labels)
#
#             acc_list.append(acc.numpy())
#             loss_list.append(loss.numpy())
#         acc_mean = np.array(acc_list).mean()
#         acc_mean_list.append(acc_mean)
#         loss_mean = np.array(loss_list).mean()
#         loss_mean_list.append(loss_mean)
#
#         x.append(epoch)
#         print(f'epoch-{epoch}, loss_mean: {loss_mean}, acc_mean: {acc_mean}\n')
#
#         paddle.save(model.state_dict(), f'./output/LeNet-Adam/{learn_rate}/{batch_size}/model_{epoch}')
#         if best_result < acc_mean:
#             best_result = acc_mean
#             best_epoch = epoch
#             paddle.save(model.state_dict(), f'./output/LeNet-Adam/{learn_rate}/{batch_size}/best_model')
#
#     print(train_lost_mean_list)
#     print(acc_mean_list)
#     print(loss_mean_list)
#     print(f'best epoch: {best_epoch}, result: {best_result}')
#
#     return train_lost_mean_list, acc_mean_list, loss_mean_list, best_result, best_epoch


def collect_eval_result():
    arr = []
    for model_dir in os.listdir('./output'):
        res = re.match(r'^(.*?)-(.*?)-(.*?)-(.*?)$', model_dir)

        if res is None:
            continue
        model_name, opt_name, learning_rate, batch_size = res[1], res[2], res[3], res[4]
        loss, acc = None, None
        with open(f'./output/{model_dir}/eval_res.txt', 'r') as f:
            s = f.read()
            res1 = re.match(r"^\{'loss': \[(.*?)], 'acc': (.*?)}$", s)
            loss, acc = res1[1], res1[2]
        print(model_name, opt_name, learning_rate, batch_size, loss, acc)
        arr.append({
            'model_name': model_name,
            'opt_name': opt_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'loss': loss,
            'acc': acc
        })
    df = pd.DataFrame(arr)
    df.to_csv('./eval_result.csv')


if __name__ == '__main__':
    collect_eval_result()
    # train(0.01, 64)