import numpy as np


# 交叉熵损失
# yp: 预测   y: groundtruth   N: 样本数
# 二分类: Loss = -(y - log(yp) + (1 - y) * log(1 - yp))
# 多分类: Loss = -sigma(yi * log(ypi)) / N
class CrossEntropyLoss():
    def __init__(self):
        self.x = None
        self.labels = None
    
    def __call__(self, x ,labels):
        self.x = x
        self.labels = labels
        return self.forward(self.x, self.labels)
    
    def forward(self, x, labels):
        cross_entropy_loss = np.sum(-(labels * np.log(x)), axis = 1).mean()
        return cross_entropy_loss

if __name__ == "__main__":
    
    # Batch: 3   Class: 4
    x = np.array([  [0.1, 0.5, 0.3, 0.1],
                    [0.2, 0.2, 0.5, 0.1],
                    [0.05, 0.1, 0.1, 0.75]])

    labels = np.array([ [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    loss_fn = CrossEntropyLoss()

    loss = loss_fn(x, labels)
    print("Cross-Entropy Loss:", loss)