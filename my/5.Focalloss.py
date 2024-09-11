import numpy as np


def FocalLoss(label, pred, 
               class_weights = None, alpha=0.5, gamma=2):
    
    p = np.where(label == 1, pred, 1 - pred)
    alpha_t = np.where(label == 1, alpha, 1 - alpha)

    focal_loss = -alpha_t * (1 - p) ** gamma * np.log(p)

    if class_weights is None:
        focal_loss = np.mean(focal_loss)
    else:
        # mutiply: 逐元素相乘
        focal_loss = np.sum(np.multiply(focal_loss, class_weights))

    return focal_loss

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.4, 0.7, 0.2])

loss = FocalLoss(y_true, y_pred)

print(loss)