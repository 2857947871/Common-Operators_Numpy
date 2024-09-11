import numpy as np


class Img2ColIndices():
    """
    卷积网路的滑动计算实际上是将feature map转换为矩阵乘法的方式
    forward前需要将feature map转换为cols格式, 每一次滑动的窗口作为cols的一列
    """
    """
    eg:
        img:
            1   2  3  4
            5   6  7  8
            9  10 11 12
            13 14 15 16
        
        kernel:
            1/4 1/4 
            1/4 1/4
            
        窗口1: 1 2 5 6
        窗口2: 2 3 6 7
        窗口3: 3 4 7 8
            .
            .
            .
        窗口8: 10 11 14 15
        窗口9: 11 12 15 16
        
        展开成列向量:
            img:
                1  2  3  5  6  7  9  10 11
                2  3  4  6  7  8  10 11 12
                5  6  7  9  10 11 13 14 15
                6  7  8  10 11 12 14 15 16
            
            kernel:
                1/4 1/4 1/4 1/4
                
            res:
                [1, 9] -> [3, 3]
    """
    def __init__(self, kernel_size, padding, stride):
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride
    
    # 将feature map按照卷积核的大小转换为适合矩阵乘法的形式
    def get_img2col_indices(self, out_h, out_w):
        
        # eg: kernel_size: 2 x 2   feature map: 4 x 4
        # i0: np.repeat(np.arange(2), 2)
        #   [0, 0, 1, 1]
        # i1: np.repeat(np.arange(4), 4) * self.stride
        #   [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] * stride(步幅: 实际位置)
        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i1 = np.repeat(np.arange(out_h), out_w) * self.stride
        i  = i0.reshape(-1, 1) + i1
        i = np.tile(i, [self.c_x, 1])

        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size)
        j1 = np.tile(np.arange(out_w), out_h) * self.stride
        j = j0.reshape(-1, 1) + j1
        j = np.tile(j, [self.c_x, 1])
        
        k = np.repeat(np.arange(self.c_x), self.kernel_size * self.kernel_size).reshape(-1, 1)
        
        # k: 通道索引   i: 高度索引   j: 宽度索引
        return k, i, j
    
    def img2col(self, x):
        self.n_x, self.c_x, self.h_x, self.w_x = x.shape
        out_h = int((self.h_x + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((self.w_x + 2 * self.padding - self.kernel_size) / self.stride + 1)

        # padding
        x_padded = None
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x


        self.img2col_indices = self.get_img2col_indices(out_h, out_w)
        k, i, j = self.img2col_indices

        # shape[batch_size, kernel_size*kernel_size*n_channel, h_out*w_out]
        cols = x_padded[:, k, i, j]
        
        return cols.transpose(1, 2, 0).reshape(
            self.kernel_size * self.kernel_size * self.c_x, -1)
        
    def col2img(self, cols):
        # 将col还原成img2col的输出shape
        cols = cols.reshape(self.kernel_size * self.kernel_size * self.c_x, -1, self.n_x)
        cols = cols.transpose(2, 0, 1)  # 调整维度顺序, 使 batch 在前面

        h_padded, w_padded = self.h_x + 2 * self.padding, self.w_x + 2 * self.padding
        x_padded = np.zeros((self.n_x, self.c_x, h_padded, w_padded))
        count = np.zeros_like(x_padded)

        # 获取img2col时的索引k, i, j
        k, i, j = self.img2col_indices

        # 使用np.add.at进行累加操作, 处理重叠区域
        np.add.at(x_padded, (slice(None), k, i, j), cols)
        np.add.at(count, (slice(None), k, i, j), 1) # 在同样位置对计数矩阵进行累加

        # 归一化处理，避免重叠区域的累加效应
        x_padded /= count

        if self.padding == 0:
            return x_padded
        else:
            return x_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]


if __name__ == "__main__":
    
    img2col = Img2ColIndices(kernel_size=2, padding=0, stride=1)
    
    x = np.array([[[[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9,  10, 11, 12],
                    [13, 14, 15, 16]]]])
    
    cols = img2col.img2col(x)
    
    """
    注: 
    [[[[ 1.  4.  6.  4.]
        [10. 24. 28. 16.]
        [18. 40. 44. 24.]
        [13. 28. 30. 16.]]]]
        当这些滑动窗口被展平为列矩阵(通过 img2col), 
        然后再还原为图像时, 重叠的区域会有多次累加操作
        像素 1: 没有重叠, 值保持 1
        像素 2: 出现在两个窗口中, 值变为 2+2 = 4
        像素 6: 出现在四个窗口中, 值变为 6+6+6+6 = 24
        像素 7: 出现在两个窗口中, 值变为 7+7 = 14
    解决方案:
        对重叠区域进行平均或其他处理
        你可以在 np.add.at 操作完成后, 使用一个归一化因子来对累加后的图像进行处理
    """
    img  = img2col.col2img(cols)
    
    print(cols)
    print(img)