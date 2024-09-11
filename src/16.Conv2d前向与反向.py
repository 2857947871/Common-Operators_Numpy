import numpy as np
from Img2Col import Img2ColIndices


class Conv2d():
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride

        # 初始化参数
        # kernel: NCHW [out_channel, in_channel, kernel_size, kernel_size]
        self.W = np.random.randn(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size) / np.sqrt(self.out_channel / 2.)
        self.b = np.zeros((self.out_channel, 1))

        self.params = [self.W, self.b]

    def __call__(self, x):
        self.n_x, _, self.h_x, self.w_x = x.shape
        self.h_out = (self.h_x + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.w_out = (self.w_x + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.img2col_indices = Img2ColIndices(self.kernel_size, self.padding, self.stride)
        
        return self.forward(x)

    def forward(self, X):
        # 将X转换成col
        self.x_col = self.img2col_indices.img2col(X)
        
        # 转换参数W的形状，使它适合与col形态的x做计算
        self.w_row = self.W.reshape(self.out_channel, -1)
        
        # 计算前向传播
        out = np.matmul(self.w_row, self.x_col) + self.b
        out = out.reshape(self.out_channel, self.h_out, self.w_out, self.n_x)
        out = out.transpose(3, 0, 1, 2)
        
        return out
    
    def backward(self, d_out):

        # 转换d_out的形状
        d_out_col = d_out.transpose(1, 2, 3, 0)
        d_out_col = d_out_col.reshape(self.out_channel, -1)
        
        d_w = np.matmul(d_out_col, self.x_col.T)
        d_w = d_w.reshape(self.W.shape)  # shape=(n_filter, d_x, h_filter, w_filter)
        d_b = d_out_col.sum(axis=1).reshape(self.out_channel, 1)
        

        d_x = self.w_row.T @ d_out_col
        # 将col态的d_x转换成image格式
        d_x = self.img2col_indices.col2img(d_x)
        
        return d_x, [d_w, d_b]

# 测试代码
if __name__ == "__main__":
    np.random.seed(0)

    # 输入数据 (1, 1, 4, 4) 的图像
    x = np.array([[[[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9,  10, 11, 12],
                    [13, 14, 15, 16]]]])

    # 卷积层参数
    in_channel = 1
    out_channel = 3
    kernel_size = 2
    padding = 0
    stride = 1

    # 创建卷积层实例
    conv = Conv2d(in_channel, out_channel, kernel_size, padding, stride)

    # 前向传播
    out = conv(x)
    print("Forward Output:")
    print(out)

    # 生成随机的梯度
    d_out = np.random.randn(*out.shape)

    # 反向传播
    d_x, grads = conv.backward(d_out)
    print("\nBackward Output:")
    print("d_x:")
    print(d_x)
    print("Gradients:")
    print("d_w:")
    print(grads[0])
    print("d_b:")
    print(grads[1])