import numpy as np


class Flatten():
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.x = x
        self.x_shape = x.shape
        
        return self.forward(x)

    def forward(self, x):
        
        # ravel(): 将多维数组展成一维数组
        # [N, C, H, W] - [N*C*H*W] -> [N , C*H*W]
        output = x.ravel().reshape(self.x_shape[0], -1)
        return output
    
    def backward(self, d_out):
        d_x = d_out.reshape(self.x_shape)
        return d_x


if __name__ == "__main__":

    x = np.random.rand(2, 3, 4, 4)

    flatten = Flatten()

    # 前向传播
    flattened_output = flatten(x)
    print("Flattened output shape:", flattened_output.shape)

    # 反向传播
    d_out = np.random.rand(*flattened_output.shape)
    reconstructed_input = flatten.backward(d_out)
    print("Reconstructed input shape:", reconstructed_input.shape)

    # 验证重建后的输入是否与原始输入形状相同
    print("Input shape is correctly reconstructed:", reconstructed_input.shape == x.shape)