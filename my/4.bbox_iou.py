import numpy as np


def IoU(boxA, boxB):
    
    # 坐标零点: 左上角
    # boxA(x1, y1, x2, y2)
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = max(boxA[2], boxA[2])
    y2 = max(boxA[3], boxA[3])

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxABrea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    interArea = (x1 - x2) * (y2 - y1)

    iou = interArea / boxAArea + boxABrea - interArea

    return iou


def IoU_numpy(boxsA, boxsB):

    # 批量处理
    # boxsA: (n, 4)   boxsB: (m, 4)
    boxsAArea = (boxsA[:, 2] - boxsA[:, 0]) * (boxsA[:, 3] - boxsA[:, 1])
    boxsBArea = (boxsB[:, 2] - boxsB[:, 0]) * (boxsB[:, 3] - boxsB[:, 1])

    # A: (n, 4) -> (n, 1, 2)
    # B: (m, 2)
    # 广播 -> (n, m, 2)
    lt = np.maximum(boxsA[:, None, :2], boxsB[:, :2])
    rb = np.maximum(boxsA[:, None, 2:], boxsB[:, 2:])

    # wh: (n, m, 2)
    wh = np.clip(rb - lt, a_min=0, a_max=None)

    # inter: (n, m)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = boxsAArea[:, None] + boxsBArea - inter

    iou = inter / union

    return iou



