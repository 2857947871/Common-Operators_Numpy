import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):

    # 防止分布过于极端
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)

    return exp_x / np.sum(exp_x)