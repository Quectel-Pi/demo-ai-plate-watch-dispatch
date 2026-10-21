#!/usr/bin/env python3
"""板上双模型全链路测试：检测(四角点) -> 透视校正 -> LPRNet 识别。

用法: python3 plate_pipeline_test.py <det.rknn> <lpr.rknn> <img...>
板上依赖: numpy cv2（系统自带）+ rknn_backend.py + pose_decode.py（同目录）
"""
import sys
import time

import cv2
import numpy as np

from rknn_backend import RKNNRuntime
from pose_decode import decode_pose

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']
BLANK = len(CHARS) - 1
DST = np.float32([[0, 0], [93, 0], [93, 23], [0, 23]])  # 94x24 TL TR BR BL


def ctc_greedy(preds):
    """preds: [68, T] -> 文本"""
    seq = np.argmax(preds, axis=0)
    out, prev = [], None
    for c in seq:
        c = int(c)
        if c != prev and c != BLANK:
            out.append(CHARS[c])
        prev = c
    return ''.join(out)


def letterbox(img, size=640):
    """RGB canvas + 还原参数。返回 (canvas_rgb, s, px, py)"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas[:, :, ::-1], s, px, py  # BGR -> RGB


def main():
    det_path, lpr_path, imgs = sys.argv[1], sys.argv[2], sys.argv[3:]
    det = RKNNRuntime(det_path)
    lpr = RKNNRuntime(lpr_path)
    print(f"[ok] det out={det._out_shape} lpr out={lpr._out_shape}")

    t_det = t_lpr = 0.0
    n = 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] {path}")
            continue
        canvas, s, px, py = letterbox(img)
        t0 = time.perf_counter()
        out = det.run(canvas.astype(np.float32).reshape(1, 640, 640, 3))
        boxes, confs, kpts = decode_pose(out, conf_thres=0.3, iou_thres=0.45)
        t_det += time.perf_counter() - t0
        n += 1
        print(f"--- {path}: {len(boxes)} 检出")
        for box, conf, k in zip(boxes, confs, kpts):
            kp = (k[:, :2] - [px, py]) / s          # 映射回原图
            kconf = k[:, 2]
            if kconf.min() < 0.3:
                print(f"    conf={conf:.2f} 角点置信不足({kconf.round(2)}) 跳过")
                continue
            M = cv2.getPerspectiveTransform(kp.astype(np.float32), DST)
            plate = cv2.warpPerspective(img, M, (94, 24))
            t1 = time.perf_counter()
            o = lpr.run(plate.astype(np.float32).reshape(1, 24, 94, 3))
            t_lpr += time.perf_counter() - t1
            text = ctc_greedy(o.reshape(o.shape[-2:]))
            print(f"    车牌={text} det_conf={conf:.2f} kpt={kp.astype(int).tolist()}")
            cv2.imwrite(f"{path}_plate_{text}.jpg", plate)

    if n:
        print(f"[耗时] det={t_det/n*1000:.1f}ms lpr={t_lpr/n*1000:.1f}ms (n={n})")
    det.close()
    lpr.close()


if __name__ == '__main__':
    main()
