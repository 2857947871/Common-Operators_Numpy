import torch
import torch.nn as nn
import numpy as np

class my_BN:
    def __init__(self, num_features, momentum=0.9, eps=1e-5, lr=0.01):
        self.running_mean = 0
        self.running_var  = 1
        self.momentum     = momentum
        self.eps          = eps
        self.beta         = 0
        self.gamma        = 1
        self.lr            = lr

    def forward(self, x, train=True):
        if train:
            self.x_mean = np.mean(x, axis=0)
            self.x_var  = np.var(x, axis=0)

            # self.running: 训练过程中的累积 mean 和 var
            self.running_mean = (1 - self.momentum) * self.x_mean + self.momentum * self.running_mean
            self.running_var  = (1 - self.momentum) * self.x_var  + self.momentum * self.running_var
            self.x_normalized  = (x - self.x_mean) / (np.sqrt(self.x_var + self.eps))
        else:
            # 推理时使用 self.running_ 保证模型对任意输入的一致性
            self.x_normalized  = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
        
        return self.gamma * self.x_normalized + self.beta

    def backward(self, dout):
        # dout: 上一层传来的梯度
        N = dout.shape[0]
        
        # 计算相对于 gamma 和 beta 的梯度
        # output = self.gamma * self.x_normalized + self.beta
        # doutput/dgama = x_normalized
        # doutput/dbeta = 1
        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dbeta  = np.sum(dout, axis=0)

        # 计算相对于输入 x 的梯度
        # dout/dx_normalized = self.gamma
        dx_normalized = dout * self.gamma

        # self.x_normalized  = (x - self.x_mean) / (np.sqrt(self.x_var + self.eps))
        # doutput/dvar = dout/dx_normalized * dx_normalized/dvar
        dvar = np.sum(dx_normalized * (self.x_normalized * -0.5 / (self.x_var + self.eps)), axis=0)

        # doutput/dmean = dout/dx_normalized * dx_normalized/dmean + doutput/dvar * dvar/dmean
        # var = sum(x - mean) ** 2 / N
        # dvar/dmean = -2 * np.sum(x - mean) / N = -2 * np.mean(x - mean)
        # 此时的 x 已经是归一化后的 x_normalized
        # dvar/dmean = -2 * np.mean(x_normalized - mean)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.x_var + self.eps), axis=0) + dvar * np.mean(-2 * (self.x_mean - self.x_normalized), axis=0)

        # doutput/dx = dout/dx_normalized * dx_normalized/dx
        # dvar/dx = 2 * (x - mean) / N
        # 此时的 x 已经是归一化后的 x_normalized
        # doutput/dx = dout/dvar  * dvar/dx
        # dmean/dx = 1 / N
        # doutput/dx = dout/dmean * dmean/dx
        dx = dx_normalized / np.sqrt(self.x_var + self.eps) + dvar * 2 * (self.x_mean - self.x_normalized) / N + dmean / N

        # 更新
        self.gamma -= self.lr * dgamma
        self.beta -= self.lr * dbeta

        return dx


x = np.random.randn(10, 5)
bn = my_BN(num_features=5, momentum=0.9, eps=1e-5)
output = bn.forward(x)

print(output)


x_torch = torch.tensor(x, dtype=torch.float32)
bn_torch = nn.BatchNorm1d(num_features=5, momentum=0.9, eps=1e-5)
with torch.no_grad():
    bn_torch.weight.fill_(1.0)  # gamma
    bn_torch.bias.fill_(0.0)    # beta

output_torch = bn_torch(x_torch).detach().numpy()
print(output_torch)


# 比较输出
print("Difference between custom and PyTorch outputs during training:", np.abs(output - output_torch).max())