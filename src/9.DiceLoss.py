# DiceLoss是一种衡量样本相似度的指标, 常用于语义分割任务, 尤其在处理前景背景不平衡问题时
# 它优化F1分数, 适用于多分类分割, 并能缓解样本不平衡的影响
# DiceLoss的计算公式如下:
# DiceLoss = 1 - (2 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum()

    return 1 - (2 * intersection + smooth) / (union + smooth)

import torch

# 定义测试函数
def test_dice_loss():
    # 创建简单的y_true和y_pred
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float32)

    # 手动计算Dice损失
    smooth = 1e-5
    intersection = 1.7  # (1*0.9 + 1*0.8 + 0*0.2 + 0*0.1)
    union = 4.0  # (1+0.9 + 1+0.8 + 0+0.2 + 0+0.1)
    expected_loss = 1 - (2 * intersection + smooth) / (union + smooth)
    
    # 使用dice_loss函数计算损失
    loss = dice_loss(y_true, y_pred)

    # 打印结果进行对比
    print(f"Expected loss: {expected_loss}")
    print(f"Dice loss: {loss.item()}")


if __name__ == "__main__":
    test_dice_loss()