# focalloss: 不同的权重来调整 loss

import numpy as np

def multiclass_focal_log_loss(y_true, y_pred, 
                                class_weights = None, alpha = 0.5, gamma = 2):
    
    # 条件表达式: y_true == 1 : y_pred ? 1 - y_pred
    # y_true 中是 1 的位置, pt 会选取 y_pred 中对应位置的值
    # y_true 中不是 1 的位置, pt 会选取 1 - y_pred 中对应位置的值
    # y_true = np.array([1, 0, 1, 0])
    # y_pred = np.array([0.8, 0.4, 0.7, 0.2])
    # pt = ([0.8, 0.6, 0.7, 0.8])
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # 带公式
    focal_loss = -alpha_t * (1 - pt) ** gamma * np.log(pt)

    if class_weights is None:
        focal_loss = np.mean(focal_loss)
    
    else:
        # mutiply: 逐元素相乘
        focal_loss = np.sum(np.multiply(focal_loss, class_weights))
    
    return focal_loss

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.4, 0.7, 0.2])

loss = multiclass_focal_log_loss(y_true, y_pred)

print(loss)