# softnms:
# 通过减少相邻重叠边界框的置信度来抑制这些边界框，而不是直接移除它们


import numpy as np

# A scores: 0.8
# B scores: 0.7
# C scores: 0.9
# IoU(A, B) = 0.99
# i = 0 
# max = A
# j = 1
# B 的 score = score * decay(IoU=0.99 -> decay 接近 0) -> 0.001 = 0.0007(通过减少相邻重叠边界框的置信度来抑制这些边界框，而不是直接移除它们)
# j = 2
# C 的 score = score * devay(IoU=0 -> decay 为 1) -> 0.9
# max = C
# 交换位置: [A, B, C] -> [C, B, A]

# i = 1
# max = B
# B 的 score < score_thresh continue

# i = 2
# max = A
# for j in range(i + 1, len(scores)) -> 不会进行循环

# [C, B, A] -> scores > score_thresh -> [C, A]

# 特点: 没有硬性的指定 IoU 阈值
def soft_nms(bboxes, scores, sigma=0.5, score_thresh=0.001):
    
    # 提坐标和面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 选出框 A 与其余所有框进行比较
    for i in range(len(scores)):
        max_idx = i
        max_score = scores[i]

        # 与其他边界框计算 IoU, 并更新置信度
        for j in range(i + 1, len(scores)):
            if scores[j] > score_thresh:
                x11 = np.maximum(x1[i], x1[j])
                y11 = np.maximum(y1[i], y1[j])
                x22 = np.minimum(x2[i], x2[j])
                y22 = np.minimum(y2[i], y2[j])
                w = np.maximum(0, x22 - x11 + 1)
                h = np.maximum(0, y22 - y11 + 1)
                overlaps = w * h
                iou = overlaps / (areas[i] + areas[j] - overlaps)

                # IoU 越大(重合的越多 -> 本质是一个框)，衰减越显著
                decay = np.exp(-(iou * iou) / sigma)
                scores[j] = scores[j] * decay

                # 保留置信度最高的边界框
                if scores[j] > max_score:
                    max_idx = j
                    max_score = scores[j]

        # 交换置信度最高的边界框和当前边界框的位置
        bboxes[i], bboxes[max_idx] = bboxes[max_idx], bboxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]

    # 过滤置信度低于阈值的边界框
    selected_idx = np.where(scores > score_thresh)
    bboxes = bboxes[selected_idx]
    scores = scores[selected_idx]

    return bboxes, scores