def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum()

    return 1 - (2 * intersection + smooth) / (union + smooth)


import torch

def test_dice_loss():

    # 创建简单的y_true和y_pred
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float32)

    # 使用dice_loss函数计算损失
    loss = dice_loss(y_true, y_pred)

    print(loss.item()) # tensor -> 标量

if __name__ == "__main__":
    test_dice_loss()