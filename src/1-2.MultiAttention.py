from math import sqrt

import torch
import torch.nn as nn

# 多头注意力机制: 
# 为什么多头: 单头自注意力机制没有可以学习的参数
# q, k, v -> linear -> 投影至较低维度 1 / head_num
# 进行注意力计算 -> 拼接

# model_dim: 模型维度, 输入和输出的向量维度
# model_dim: NLP -> 词向量
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads # 每个头的维度

        assert model_dim % num_heads == 0, "model_dim 必须整除注意力头的数量"

        self.linear = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim = -1)
        
    def attention(self, q, k, v, head_dim):
        scores = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim))
        output = torch.matmul(scores, v)
        
        return output

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        
        # 分割多头
        q = self.linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(2, 3)
        k = self.linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(2, 3)
        v = self.linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(2, 3)
        
        scores = self.attention(q, k, v,self.head_dim)
        
        # 重新调整大小(和输入同)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        output = self.linear(concat)
        
        return output


if __name__ == "__main__":
    q = torch.randn(1, 64, 64)
    k = torch.randn(1, 64, 64)
    v = torch.randn(1, 64, 64)
    
    attention = MultiHeadSelfAttention(model_dim=64, num_heads=8)
    output = attention(q, k, v)

    print(output)