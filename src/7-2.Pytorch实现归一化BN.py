import torch
import torch.nn as nn
import numpy as np


class my_BN:
    def __init__(self, num_features, eps=1e-5, lr=0.001, momentum=0.9):
        self.running_mean = 0
        self.running_var  = 1
        self.momentum     = momentum
        self.eps          = eps
        self.beta         = 0
        self.gamma        = 1
        self.lr            = lr

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


x = torch.randn(10, 5)
bn = my_BN(5)
output = bn.forward(x)
print(output)

bn_torch = nn.BatchNorm1d(num_features=5, momentum=0.9, eps=1e-5)
with torch.no_grad():
    bn_torch.weight.fill_(1.0)  # gamma
    bn_torch.bias.fill_(0.0)    # beta

output_torch = bn_torch(x).detach().numpy()
print(output_torch)


# 比较输出
print("Difference between custom and PyTorch outputs during training:", np.abs(output - output_torch).max())