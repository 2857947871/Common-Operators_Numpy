import numpy as np


# sigmoid = 1 / (1 + exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sofrmax = exp(x) / sum(exp(x))
def softmax(x):
    
    # 防止分布过于极端, 减去最大值
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    
    return exp_x / np.sum(exp_x)