import numpy as np


class SGD():
    def __init__(self, parameters, lr, momentum=None):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        
        if self.momentum is not None:
            self.velocity = self.velocity_inital()
    
    def __call__(self, grads):
        self.grads = grads
        return self.updata_parameters(self.grads)
    
    def updata_parameters(self, grads):
        if self.momentum is None:
            
            # zip: 一起遍历
            for param, grad in zip(self.parameters, grads):
                param -= self.lr * grad
        else:
            # 首次更新: velocity为0, 没起作用, 与普通SGD相同
            # 后续更新: velocity积累了之前的动量, 因此参数更新会根据前几次的梯度变化进行调整
            for i in range(len(self.parameters)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * self.grads[i]
                self.parameters[i] += self.velocity[i]
    
    def velocity_inital(self):
        velocity = []
        
        # velocity: 速度变量, 存储每个参数的动量
        # velocity: 初始化成与param维度相同的全0矩阵
        for param in self.parameters:
            velocity.append(np.zeros_like(param))
        
        return velocity


if __name__ == "__main__":
    
    np.random.seed(42)
    X = np.random.rand(10, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(10) * 0.1  # 真实的关系是 y = 3 * x + 2

    W = np.random.randn(1)
    b = np.random.randn(1)

    # 学习率和动量
    lr = 0.1
    momentum = 0.9

    # 初始化 SGD 优化器
    # 传入param
    optimizer = SGD(parameters=[W, b], lr=lr, momentum=momentum)

    # y = kx + b
    # 自行broadcast
    def forward(x):
        return np.dot(x, W) + b

    def mse_loss(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    # 训练
    for epoch in range(100):
        # 前向传播
        y_pred = forward(X)
        
        # 计算损失
        loss = mse_loss(y_pred, y)

        # 计算梯度
        grad_W = 2 * (np.dot(X.T, (y_pred - y))) / len(y)
        grad_b = 2 * (y_pred - y).mean()
        
        # 更新参数
        # 传入grads
        optimizer([grad_W, grad_b])
        
        # 每10个epoch打印一次损失和参数
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss}, W = {W[0]}, b = {b[0]}")

    # 打印最终的参数
    print(f"Final parameters: W = {W[0]}, b = {b[0]}")
