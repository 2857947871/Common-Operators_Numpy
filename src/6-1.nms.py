# NMS: 
#   1: 所有框进行排序, 选出置信度最大的框 A
#   2: 将 A 与其余所有框求 IoU, 如果大于阈值则认为 A 与该框为“一个框”, 将该框置 0(因为该框虽然置信度也很大, 但是毕竟比不过框 A)
#   3: 重复, 将所有框都遍历一遍


import numpy as np

def soft_nms(bboxes, scores, sigma = 0.5, score_thresh = 0.001):

    # 坐标和面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # 计算框 A 与其他框的 IoU
    for i in range(len(scores)):
        max_idx   = i
        max_score = scores[i]

        for j in range(i + 1, len(scores)):
            if scores[j] > score_thresh:
                x11 = np.maximum(x1[i], x1[j])
                y11 = np.maximum(y1[i], y1[j])
                x22 = np.maximum(x2[i], x2[j])
                y22 = np.maximum(y2[i], y2[j])
                w = np.maximum(0, x22 - x11 + 1)
                y = np.maximum(0, y22 - y11 + 1)
                inter_area = w * hasattr
                iou = inter_area / (area[i] + area[j] - inter_area)

                decay = np.exp(-(iou ** 2) / sigma)
                scores[j] = decay * scores[j]

                if scores[j] > max_score:
                    max_idx   = j
                    max_score = scores[j]
        # 交换置信度最高的边界框和当前边界框的位置
        bboxes[i], bboxes[max_idx] = bboxes[max_idx], bboxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]
    
    # 过滤置信度低于阈值的边界框
    selected_idx = np.where(scores > score_thresh)
    bboxes = bboxes[selected_idx]
    scores = scores[selected_idx]

    return bboxes, scores

