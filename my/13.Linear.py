import numpy as np

class Linear():
    def __init__(self, dim_in, dim_out):
        scale       = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
        self.bias   = np.random.standard_normal(dim_out) / scale
        self.params = [self.weight, self.bias]

    def __call__(self, x):
        self.x = x
        return self.forward()

    def forward(self):
        return np.dot(self.x, self.weight) + self.bias
    
    def backward(self, d_out):
        
        # 详细见noteability
        # y = kx + b
        # dy/dx = k
        # dy/dw = x
        # dy/db = 1
        d_x = np.dot(d_out, self.weight.T)
        d_w = np.dot(self.x.T, d_out)
        d_b = np.mean(d_out, axis=0)
        
        return d_x, [d_w, d_b]

if __name__ == "__main__":
    # 定义输入和输出维度
    dim_in = 3
    dim_out = 2

    # 创建一个 Linear 层的实例
    linear_layer = Linear(dim_in, dim_out)

    # 创建一个输入向量
    x = np.array([[0.5, -1.2, 3.3]])

    # 进行前向传播
    output = linear_layer(x)
    print("Forward Output:")
    print(output)

    # 假设我们有一个从上层传递下来的梯度（d_out），这里随机生成一个
    d_out = np.random.standard_normal((1, dim_out))

    # 进行反向传播
    d_x, grads = linear_layer.backward(d_out)
    d_w, d_b = grads

    print("\nBackward Output:")
    print("d_x:")
    print(d_x)
    print("d_w:")
    print(d_w)
    print("d_b:")
    print(d_b)