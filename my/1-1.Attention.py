import numpy as np

import torch
import torch.nn as nn

# 注意力机制
# softmax((q * k^T) / scale) = attention_scores
# output = attention_scores * v
class Attention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # q, k, v: [B, W, H]
        attention_scores = self.softmax((torch.matmul(q, k.transpose(1, 2))) / self.scale)
        
        output = torch.matmul(attention_scores, v)
        
        return attention_scores, output


if __name__ == "__main__":
    q = torch.randn(1, 64, 64)
    k = torch.randn(1, 64, 64)
    v = torch.randn(1, 64, 64)
    
    attention = Attention(scale=5)
    attention_scores, output = attention(q, k, v)
    
    print(attention_scores)
    print(output)