#!/usr/bin/env python3
"""YOLOv8/11-pose 板端解码：conf 过滤 + NMS + kpt 提取（不依赖 torch）。

ONNX/RKNN 输出布局: [1, 4+nk*3, N] = cxcywh(已decode,输入尺度) + sigmoid conf
+ kpt(x,y 已decode, sigmoid conf)。N=8400(640 输入)。
"""
import numpy as np


def nms_xyxy(boxes, scores, iou_thres=0.45):
    """纯 numpy NMS。boxes: [N,4] xyxy, scores: [N] -> keep idx"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def decode_pose(out, conf_thres=0.3, iou_thres=0.45, max_det=20):
    """out: [1, 4+nk*3, N] 或 [4+nk*3, N] -> (boxes_xyxy[N], confs[N], kpts[N,K,3])

    kpts x,y 为输入尺度绝对坐标，conf 为 sigmoid 后。
    """
    pred = out.reshape(out.shape[-2], -1) if out.ndim == 3 else out
    nk = (pred.shape[0] - 5) // 3
    conf = pred[4]
    mask = conf > conf_thres
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty((0, nk, 3))
    cand = pred[:, mask]                        # (5+3K, M)
    cx, cy, bw, bh = cand[0], cand[1], cand[2], cand[3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2,
                      cx + bw / 2, cy + bh / 2], axis=1)
    scores = conf[mask]
    kpts = cand[5:].T.reshape(-1, nk, 3)         # (M, K, 3)

    keep = nms_xyxy(boxes, scores, iou_thres)[:max_det]
    return boxes[keep], scores[keep], kpts[keep]
