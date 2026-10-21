#!/usr/bin/env python3
"""板上全链路 v2：HyperLPR y5fu 检测（四角点+层数） -> 透视校正 -> LPRNet 识别。

用法: python3 hlpr_pipeline_test.py <det.rknn> <lpr.rknn> <img...>
板上依赖: numpy cv2 + rknn_backend.py（同目录）
"""
import sys
import time

import cv2
import numpy as np

from rknn_backend import RKNNRuntime

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']
BLANK = len(CHARS) - 1
DST = np.float32([[0, 0], [93, 0], [93, 23], [0, 23]])  # TL TR BR BL -> 94x24


def ctc_greedy(preds):
    seq = np.argmax(preds, axis=0)
    out, prev = [], None
    for c in seq:
        c = int(c)
        if c != prev and c != BLANK:
            out.append(CHARS[c])
        prev = c
    return ''.join(out)


def letter_box(img, size=320):
    """复刻 hyperlpr3: 等比缩放居中填充。返回 (rgb_canvas, r, left, top)"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.zeros((size, size, 3), np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return canvas[:, :, ::-1], r, left, top  # BGR -> RGB


def nms(boxes, iou_thresh):
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
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=int)


def decode_y5(out, r, left, top, conf_thresh=0.3, iou_thresh=0.5):
    """out: [1,N,15] -> [[x1,y1,x2,y2,score, kpt8(原图尺度), layer]]

    布局: 0:4=xywh(letterbox尺度) 4=obj 5:13=角点xy(letterbox尺度) 13:15=双类(乘obj)
    """
    dets = out.reshape(-1, 15)
    dets = dets[dets[:, 4] > conf_thresh]
    if not len(dets):
        return np.empty((0, 14))
    dets[:, 13:15] *= dets[:, 4:5]
    xywh = dets[:, :4]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    score = dets[:, 13:15].max(1, keepdims=True)
    layer = dets[:, 13:15].argmax(1).reshape(-1, 1)
    output = np.concatenate([boxes, score, dets[:, 5:13], layer], 1)
    output = output[nms(output, iou_thresh)]
    # 还原到原图（角点列 5:12 同步）
    output[:, [0, 2, 5, 7, 9, 11]] = (output[:, [0, 2, 5, 7, 9, 11]] - left) / r
    output[:, [1, 3, 6, 8, 10, 12]] = (output[:, [1, 3, 6, 8, 10, 12]] - top) / r
    return output


def main():
    det_path, lpr_path, imgs = sys.argv[1], sys.argv[2], sys.argv[3:]
    det = RKNNRuntime(det_path)
    lpr = RKNNRuntime(lpr_path)

    t_det = t_lpr = t_total = 0.0
    n = 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        t0 = time.perf_counter()
        canvas, r, left, top = letter_box(img)
        out = det.run(canvas.astype(np.float32).reshape(1, 320, 320, 3))
        dets = decode_y5(out, r, left, top)
        t1 = time.perf_counter()
        t_det += t1 - t0
        n += 1
        print(f"--- {path}: {len(dets)} 检出")
        for d in dets:
            kp = d[5:13].reshape(4, 2).astype(np.float32)
            M = cv2.getPerspectiveTransform(kp, DST)
            plate = cv2.warpPerspective(img, M, (94, 24))
            o = lpr.run(plate.astype(np.float32).reshape(1, 24, 94, 3))
            t_lpr += time.perf_counter() - t1
            t1 = time.perf_counter()
            text = ctc_greedy(o.reshape(68, 18))
            print(f"    车牌={text} det={d[4]:.2f} layer={int(d[13])}")
            cv2.imwrite(f"{path}_hlpr_{text}.jpg", plate)
        t_total += time.perf_counter() - t0

    print(f"[耗时] det={t_det/n*1000:.1f}ms lpr={t_lpr/n*1000:.1f}ms "
          f"全流程={t_total/n*1000:.1f}ms → {n/(t_total)*1:.1f} 图/s (n={n})")
    det.close()
    lpr.close()


if __name__ == '__main__':
    main()
