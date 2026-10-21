#!/usr/bin/env python3
"""板上 LPRNet 验证：ctypes 直调 NPU，BGR/RGB 两路对比 + CTC 解码 + 耗时统计。

用法（板上）: python3 lprnet_test.py <model.rknn> <test.jpg>
依赖: numpy cv2（板上系统自带）+ rknn_backend.py（同目录）
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


def ctc_greedy(preds):
    """preds: [n_class, T] 或 [T, n_class]，自动按 argmax 长度轴解码。"""
    n_class = len(CHARS)
    if preds.shape[0] != n_class and preds.shape[-1] != n_class:
        raise ValueError(f"unexpected shape {preds.shape}")
    if preds.shape[0] == n_class:
        seq = np.argmax(preds, axis=0)
    else:
        seq = np.argmax(preds, axis=-1)
    out, prev = [], None
    for c in seq:
        if c != prev and c != n_class - 1:  # 去重 + 去 blank（最后一类）
            out.append(CHARS[int(c)])
        prev = c
    return ''.join(out)


def main():
    model, img_path = sys.argv[1], sys.argv[2]
    rt = RKNNRuntime(model)
    print(f"[ok] n_input={rt.n_input} n_output={rt.n_output} out_shape={rt._out_shape}")

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"read {img_path} failed")
    plate = cv2.resize(img, (94, 24))
    h, w = plate.shape[:2]

    for tag, arr in (("BGR", plate), ("RGB", cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))):
        t0 = time.perf_counter()
        # 多跑几次取稳定耗时
        out = None
        for _ in range(20):
            out = rt.run(arr.astype(np.float32).reshape(1, h, w, 3))
        dt = (time.perf_counter() - t0) / 20 * 1000
        text = ctc_greedy(out.reshape(out.shape[-2:]).squeeze() if out.ndim > 2 else out)
        print(f"[{tag}] shape={out.shape} 耗时={dt:.1f}ms/帧 结果={text}")

    rt.close()


if __name__ == '__main__':
    main()
