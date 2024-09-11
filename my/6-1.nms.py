import numpy as np

def nms(bboxs, scores, iou_threas):
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        
        i = index[0]
        result.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        inter_area = w * h
        ious = inter_area / (areas[i] + areas[index[1:]] - inter_area)

        idx = np.where(ious <= iou_threas)
        index = index[idx + 1]
    bboxs, scores = bboxs[result], scores[result]

    return bboxs, scores
