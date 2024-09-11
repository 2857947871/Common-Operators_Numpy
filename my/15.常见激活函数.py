import numpy as np

# Relu
class Relu:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return self.forward(x)
    
    def forward(self, x):

        return np.maximum(0, x)
    
    def backward(self, d_out):
        
        # dy/dx = 1
        grad_relu = self.x > 0
        return grad_relu * d_out
    
class Tanh:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return self.forward(x)
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, d_out):
        grad_tanh = 1 - (np.tanh(self.x)) ** 2
        return grad_tanh * d_out

class Sigmoid:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return self.forward(x)
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, d_out):
        grad_sigmoid = self.forward(self.x) * (1 - self.forward(self.x))
        return grad_sigmoid * d_out
    

if __name__ == "__main__":
    
    # 定义输入数据
    x = np.array([-1.0, 0.0, 1.0])

    # 随机生成一个从上层传递下来的梯度（d_out）
    d_out = np.random.standard_normal(x.shape)
    
    # 验证 ReLU
    relu = Relu()
    relu_output = relu(x)
    relu_grad = relu.backward(d_out)

    print("ReLU Forward Output:")
    print(relu_output)
    print("ReLU Backward Output:")
    print(relu_grad)

    # 验证 Tanh
    tanh = Tanh()
    tanh_output = tanh(x)
    tanh_grad = tanh.backward(d_out)

    print("\nTanh Forward Output:")
    print(tanh_output)
    print("Tanh Backward Output:")
    print(tanh_grad)

    # 验证 Sigmoid
    sigmoid = Sigmoid()
    sigmoid_output = sigmoid(x)
    sigmoid_grad = sigmoid.backward(d_out)

    print("\nSigmoid Forward Output:")
    print(sigmoid_output)
    print("Sigmoid Backward Output:")
    print(sigmoid_grad)