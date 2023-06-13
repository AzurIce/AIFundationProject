import matplotlib.pyplot as plt

train_lost_mean_list = [0.2701122, 0.13185214, 0.1221525, 0.10952354, 0.10743101, 0.10057195, 0.09981425, 0.10487731, 0.10287207, 0.09623766, 0.1067107, 0.0897195, 0.10868865, 0.10616623, 0.08888833, 0.10306341, 0.1148383, 0.110541634, 0.12667134, 0.10642897]
acc_mean_list = [0.9546226, 0.9688998, 0.9682508, 0.9766873, 0.9648562, 0.97379196, 0.975639, 0.979383, 0.9798323, 0.9770367, 0.976887, 0.9779353, 0.9726937, 0.9769868, 0.97773564, 0.9770367, 0.9746905, 0.9690495, 0.9685503, 0.9774361]
loss_mean_list = [0.14865646, 0.12268578, 0.13409601, 0.09909997, 0.15146329, 0.11395644, 0.114795074, 0.09799624, 0.099047005, 0.11472169, 0.11319439, 0.11025395, 0.13326895, 0.12690876, 0.12145358, 0.10586404, 0.12062278, 0.16889642, 0.1701099, 0.17025182]
best_epoch, best_result = 8, 0.9798322916030884

if __name__ == '__main__':
    x = [i for i in range(len(acc_mean_list))]
    plt.plot(x, acc_mean_list)
    plt.plot(x, loss_mean_list)
    plt.plot(x, train_lost_mean_list)
    plt.show()
    while True:
        pass