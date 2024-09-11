from math import sqrt

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, head_nums):
        super().__init__()
        self.head_nums = head_nums
        self.model_dim = model_dim # 输入输出维度
        self.head_dims = self.model_dim // self.head_nums # 每个头的维度
        
        assert self.model_dim % self.head_nums == 0, "model_dim 必须整除注意力头的数量"
        
        # 不能省 -> q k v output 各自有各自的权重
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def attention(self, q, k, v, scale):
        scores = self.softmax(torch.matmul(q, k.transpose(2, 3)) / sqrt(scale))
        output = torch.matmul(scores, v)
        
        return scores, output

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 分割
        q = self.linear_q(x).view(batch_size, -1, self.head_nums, self.head_dims)
        k = self.linear_k(x).view(batch_size, -1, self.head_nums, self.head_dims)
        v = self.linear_v(x).view(batch_size, -1, self.head_nums, self.head_dims)
        
        scores, output = self.attention(q, k, v, self.head_nums)
        
        output = output.view(batch_size, -1, self.model_dim)
        output = self.linear_out(output)
        
        return scores, output


if __name__ == "__main__":
    q = torch.randn(1, 64, 64)
    k = torch.randn(1, 64, 64)
    v = torch.randn(1, 64, 64)
    
    attention = MultiHeadSelfAttention(model_dim=64, head_nums=8)
    scores, output = attention(q)

    print(scores)
    print(output)