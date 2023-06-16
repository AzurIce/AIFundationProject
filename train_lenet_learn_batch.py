from itertools import product

import numpy as np

from models.LeNet import LeNet

if __name__ == '__main__':
    for learning_rate, batch_size in list(product(np.arange(0.001, 0.01, 0.001), [16, 32, 64, 128, 256, 512])):
        model = LeNet()
        model.train_model('Momentum', learning_rate, batch_size)
        model.evaluate_model('Momentum', learning_rate, batch_size)