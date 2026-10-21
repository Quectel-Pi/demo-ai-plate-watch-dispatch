#!/usr/bin/env python3
"""车牌检测器：y5fu RKNN 包装（letterbox → NPU → 解码 → NMS）。

输出 det: [x1,y1,x2,y2,score, kp8..., layer]，坐标为原图尺度。
"""
import cv2
import numpy as np

from rknn_backend import RKNNRuntime


class PlateDetector:
    def __init__(self, model_path: str, lib_path: str = "librknnrt.so",
                 size: int = 640, conf_thresh: float = 0.3,
                 iou_thresh: float = 0.5):
        self.rt = RKNNRuntime(model_path, lib_path)
        self.size = size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    # ---- letterbox：等比缩放居中填充，输出 RGB canvas ----
    def _letterbox(self, bgr: np.ndarray):
        size = self.size
        h, w = bgr.shape[:2]
        r = min(size / h, size / w)
        nw, nh = round(w * r), round(h * r)
        canvas = np.zeros((size, size, 3), np.uint8)
        left, top = (size - nw) // 2, (size - nh) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
        return canvas[:, :, ::-1], r, left, top  # BGR→RGB

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """返回 [n,14]：x1,y1,x2,y2,score,kp8(x,y×4),layer（原图坐标）"""
        canvas, r, left, top = self._letterbox(bgr)
        out = self.rt.run(canvas.astype(np.float32).reshape(1, self.size, self.size, 3))
        return self._decode(out, r, left, top)

    # ---- y5fu 输出布局 [1,N,15]: xywh + obj + kp8 + 双类(乘obj) ----
    def _decode(self, out: np.ndarray, r: float, left: int, top: int):
        dets = out.reshape(-1, 15)
        dets = dets[dets[:, 4] > self.conf_thresh]
        if not len(dets):
            return np.empty((0, 14))
        dets[:, 13:15] *= dets[:, 4:5]
        xywh = dets[:, :4]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        score = dets[:, 13:15].max(1, keepdims=True)
        layer = dets[:, 13:15].argmax(1).reshape(-1, 1)
        output = np.concatenate([boxes, score, dets[:, 5:13], layer], 1)
        output = output[self._nms(output, self.iou_thresh)]
        # letterbox 坐标 → 原图坐标（bbox 与角点同步）
        output[:, [0, 2, 5, 7, 9, 11]] = (output[:, [0, 2, 5, 7, 9, 11]] - left) / r
        output[:, [1, 3, 6, 8, 10, 12]] = (output[:, [1, 3, 6, 8, 10, 12]] - top) / r
        return output

    @staticmethod
    def _nms(boxes: np.ndarray, thresh: float):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = boxes[:, 4].argsort()[::-1]
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
            order = order[1:][iou <= thresh]
        return np.array(keep, dtype=int)

    def close(self):
        self.rt.close()
