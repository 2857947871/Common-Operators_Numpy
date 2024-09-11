import numpy as np

def IoU(boxA, boxB):

    # 坐标零点: 左上角 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou

def IoU_numpy(boxA, boxB):

    # 批量处理, 所有框的 [2] [0] [3] [1]
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # 左上角和右下角
    # [:, :2]: (m, 4)
    # [:, None, :2]: 增加一个维度, (n, 4) -> (n, 1, 4)
    # [:2]: 从索引 2 开始(不包括 2), 提取前两个 -> 直接提取右上角
    # [2:]: 从索引 2 开始(不包括 2), 提取到末尾 -> 直接提取左上角
    # boxB[:, :2]： 所有 box 的左上角坐标
    # eg: boxA = [[2, 2, 5, 5], [1, 1, 4, 5]]   boxB = [[3, 3, 6, 6], [0, 0, 2, 2]]
    # [2, 2, 5, 5] 分别与 [3, 3, 6, 6], [0, 0, 2, 2] 比较
    # [[[3, 3], [2, 2]], [[3, 3], [1, 1]]] -> 直接返回 n * m 组左上角坐标
    lt = np.maximum(boxA[:, None, :2], boxB[:, :2])
    rb = np.minimum(boxA[:, None, 2:], boxB[:, 2:])

    # 交集
    # wh: (n, m, 2)
    # inter: (n, m)
    wh = np.clip(rb - lt, a_min=0, a_max=None)

    # wh[:, :, 0]: (n, m)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # inter: (n, m)
    # boxAArea: (n,) -> (n, 1)
    # boxBArea: (n,)
    union = boxAArea[:, None] + boxBArea - inter

    iou = inter / union

    return iou


boxA = [4, 4, 5, 5]
boxB = [3, 3, 6, 6]

boxA_np = np.array([
    [2, 2, 5, 5],  # 边界框 1
    [1, 1, 4, 4]   # 边界框 2
])

boxB_np = np.array([
    [3, 3, 6, 6],  # 边界框 A
    [0, 0, 2, 2],  # 边界框 B
    [1, 1, 2, 2]   # 边界框 C
])

print(IoU(boxA, boxB))
print(IoU_numpy(boxA_np, boxB_np))