import numpy as np
from Img2Col import Img2ColIndices

class Maxpool():
    def __init__(self, size, stride):
        '''
        和卷积类似 -> img2col
        size: maxpool的尺寸(滑动窗口)
        stride: 步长
        '''
        self.size = size
        self.stride = stride
    
    def __call__(self, x):
        # x: [batch, channels, height, width]
        self.n_x, self.c_x, self.h_x, self.w_x = x.shape
        
        # 计算输出尺寸(同卷积)
        self.out_h = int((self.h_x - self.size) / self.stride + 1)
        self.out_w = int((self.w_x - self.size) / self.stride + 1)
        
        # img2col
        self.img2col_indices = Img2ColIndices(self.size, padding=0, stride=self.stride)
        
        return self.forward(x)
    
    def forward(self, x):  # sourcery skip: inline-immediately-returned-variable
        
        # x: [N, C, H, W] -> x_col: [k_h * k_w, N * C * out_h * out_w]
        #   每个通道独立出来, 为了img2col
        x_reshaped = x.reshape(self.n_x * self.c_x, 1, self.h_x, self.w_x)
        self.x_col = self.img2col_indices.img2col(x_reshaped)
        
        '''
        x_col:
            1, 1, 4, 5
            2, 1, 3, 7
            3, 9, 1, 1
            
        max_indices:
            2, 2, 0, 1
            
        out:
            x_col[]
            
        '''
        # axis=0: k_h * k_w这个维度, 上的max
        # max_indices: 每一列最大值的索引
        self.max_indices = np.argmax(self.x_col, axis=0)
        
        #                行            列
        # eg: out: [[2, 2, 3, 1], [0, 1, 2, 3]]
        out = self.x_col[self.max_indices, range(self.max_indices.size)]
        out = out.reshape(self.out_h, self.out_w, self.n_x, self.c_x).transpose(2, 3, 0, 1)
        
        return out
    
    def backward(self, d_out):
        
        # 初始化(最大值位置梯度保留, 其余置0)
        # d_x_col: [k_h * k_w, N * C * out_h * out_w]
        d_x_col = np.zeros_like(self.x_col)
        
        # d_out: [N, C, H, W]
        # reval(): 展平
        d_out_flat = d_out.transpose(2, 3, 0, 1).ravel()
        
        # 最大值位置梯度保留, 其余置0
        # eg: d_x_col: [[2, 1, 3, 1], [0, 1, 2, 3]]
        #     d_out_flat: [d0, d1, d2, d3]   因为maxpool后仅剩4个元素
        #     d_x_col[2, 0] = d_out_flat[0]   d_x_col[2, 1] = d_out_flat[1] ...
        d_x_col[self.max_indices, range(self.max_indices.size)] = d_out_flat
        
        # col2img
        d_x = self.img2col_indices.col2img(d_x_col)
        d_x = d_x.reshape(self.n_x, self.c_x, self.h_x, self.w_x)
        
        return d_x


if __name__ == "__main__":
    # 示例输入
    x = np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8]],  # Channel 1
        [[9, 10, 11, 12], [13, 14, 15, 16]]  # Channel 2
    ])

    x = x[np.newaxis, :, :, :]  # Add batch dimension

    # 创建 Maxpool 实例
    pool = Maxpool(size = 2, stride = 2)

    # 执行前向传播
    out = pool(x)
    
    print("Maxpool Output:")
    for batch in range(out.shape[0]):
        print(f"Batch {batch + 1}:")
        for channel in range(out.shape[1]):
            print(f"  Channel {channel + 1}:")
            for row in range(out.shape[2]):
                print(f"    {out[batch, channel, row]}")

    # 假设 d_out 为反向传播中的梯度
    d_out = np.ones_like(out)

    d_x = pool.backward(d_out)
    
    print("\nGradient with respect to input x:")
    for batch in range(d_x.shape[0]):
        print(f"Batch {batch + 1}:")
        for channel in range(d_x.shape[1]):
            print(f"  Channel {channel + 1}:")
            for row in range(d_x.shape[2]):
                print(f"    {d_x[batch, channel, row]}")