import torch
import torch.nn as nn
import numpy as np

class my_BN_torch:
    def __init__(self, num_features, eps=1e-5, lr=0.001, momentum=0.9):
        self.running_mean = 0
        self.running_var = 0
        self.momentum = momentum
        self.eps = eps
        self.beta = 0
        self.gamma = 1
        self.lr = lr

    def forward(self, x, train=True):
        if train:
            batch_mean = torch.mean(x, dim=0)
            batch_var  = torch.var(x, dim=0, unbiased=False) # unbiased=False: 计算方差时除以 N 而不是 N-1

            self.running_mean = (1 - self.momentum) * batch_mean + self.momentum * self.running_mean
            self.running_var  = (1 - self.momentum) * batch_var  + self.momentum * self.running_var
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        output = self.gamma * x_normalized + self.beta

        return output

class my_BN_numpy:
    def __init__(self, num_features, eps=1e-5, lr=0.001, momentum=0.9):
        self.running_mean = 0
        self.running_var = 0
        self.momentum = momentum
        self.eps = eps
        self.beta = 0
        self.gamma = 1
        self.lr = lr
    
    def forward(self, x, train=True):
        if train:
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var  = np.var(x, axis=0)

            self.running_mean = (1 - self.momentum) * self.batch_mean + self.momentum * self.running_mean
            self.running_var  = (1 - self.momentum) * self.batch_var  + self.momentum * self.running_var
            self.x_normalized = (x - self.batch_mean) / (np.sqrt(self.batch_var + self.eps))
        else:
            self.x_normalized = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
        output = self.gamma * self.x_normalized + self.beta

        return output

    def backward(self, dout):

        N = dout.shape[0]

        # gamma和beta的梯度
        # output = self.gamma * self.x_normalized + self.beta
        # doutput/dgama = x_normalized
        # doutput/dbeta = 1
        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dbeta  = np.sum(dout * 1, axis=0)

        # x_normalized的梯度
        # x_normalized的梯度
        # output = self.gamma * self.x_normalized + self.beta
        # dout/dx_normalized = self.gamma
        dx_normalized = dout * self.gamma

        # var的梯度
        # self.x_normalized = (x - self.batch_mean) / (np.sqrt(self.batch_var + self.eps))
        # doutput/dvar = dout/dx_normalized * dx_normalized/dvar
        dvar = np.sum(dx_normalized * (self.x_normalized * -0.5 / (self.batch_var + self.eps)), axis=0)

        # mean的梯度(var中也有mean)
        # self.x_normalized = (x - self.batch_mean) / (np.sqrt(self.batch_var + self.eps))
        # doutput/dmean = dout/dx_normalized * dx_normalized/dmean + doutput/dvar * dvar/dmean
        # var = sum(x - mean) ** 2 / N
        # dvar/dmean = -2 * np.sum(x - mean) / N = -2 * np.mean(x - mean)
        # 此时的 x 已经是归一化后的 x_normalized
        # dvar/dmean = -2 * np.mean(x_normalized - mean)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.batch_var + self.eps), axis=0) + dvar * np.mean(-2 * (self.batch_mean - self.x_normalized), axis=0)

        # x的梯度
        # doutput/dx = dout/dx_normalized * dx_normalized/dx + dx_normalized/dvar * dvar/dx + dx_normalized/dmean * dmean/dx
        # dx_normalized/dx = 1 / np.sqrt(self.batch_var + self.eps)
        # dvar/dx = 2 * (x - mean) / N
        # dmean/dx = 1 / N
        dx = dx_normalized / np.sqrt(self.x_var + self.eps) + dvar * 2 * (self.x_mean - self.x_normalized) / N + dmean / N

        # 更新超参数
        self.gamma -= self.lr * dgamma
        self.beta -= self.lr * dbeta

        return dx
    

x_numpy = np.random.randn(10, 5)
BN_numpy = my_BN_numpy(num_features=5, momentum=0.9, eps=1e-5)
output = BN_numpy.forward(x_numpy)

x_torch = torch.tensor(x_numpy, dtype=torch.float32)
BN_torch = nn.BatchNorm1d(num_features=5, momentum=0.9, eps=1e-5)
with torch.no_grad():
    BN_torch.weight.fill_(1.0)  # gamma
    BN_torch.bias.fill_(0.0)    # beta

output_torch = BN_torch(x_torch).detach().numpy()


print(output_torch)

print(output)

# 比较输出
print("Difference between custom and PyTorch outputs during training:", np.abs(output - output_torch).max())