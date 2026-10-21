#!/usr/bin/env python3
"""车牌识别器：双引擎（默认 rpv3，备选 LPRNet）。

rpv3  : HyperLPR3 识别模型（48x160, 78类, blank=0），小牌/绿牌省份字更稳
LPRNet: rknn_model_zoo（94x24, 68类, blank=67）
均为 BGR 0-255 输入（RKNN 转换时已固化 (x-127.5)/127.5）。
"""
import math

import cv2
import numpy as np

from rknn_backend import RKNNRuntime

# ---- LPRNet 字符表（blank 在末位） ----
LPR_CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'I', 'O', '-']

# ---- rpv3 字符表（blank=0 在首位，' 为占位） ----
RPV3_CHARS = ['blank', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
              'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋',
              '桂', '民', '沪', '津', '浙', '渝', '港', '湘', '琼', '甘', '皖',
              '粤', '航', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂',
              '闽', '陕', '青', '鲁', '黑', '领', '使', '澳']

DST = np.float32([[0, 0], [93, 0], [93, 23], [0, 23]])  # TL TR BR BL → 94x24


class Rpv3Reader:
    """HyperLPR3 rpv3：按角点实测尺寸校正 → 等比缩放补零到 48x160 → CTC(blank=0)

    预处理严格复刻官方两步：
    1) get_rotate_crop_image：透视校正到角点实测宽高（保留原始分辨率与真实宽高比，
       蓝牌 3.14 / 绿牌 3.43，而非 94/24=3.92 的畸变拉伸）；
    2) encode_images：等比缩放到高 48，宽 ceil(48*ratio)>160 时直接拉伸到 160，
       否则缩放到 max(ceil(48*ratio),48) 宽后右侧补零到 160。
    """
    def __init__(self, model_path: str, lib_path: str = "librknnrt.so"):
        self.rt = RKNNRuntime(model_path, lib_path)
        self.chars = RPV3_CHARS

    def read(self, bgr: np.ndarray, kpt4: np.ndarray):
        kpt = kpt4.astype(np.float32)
        w = int(max(np.linalg.norm(kpt[0] - kpt[1]), np.linalg.norm(kpt[2] - kpt[3])))
        h = int(max(np.linalg.norm(kpt[0] - kpt[3]), np.linalg.norm(kpt[1] - kpt[2])))
        w, h = max(w, 2), max(h, 2)
        m = cv2.getPerspectiveTransform(kpt, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        plate = cv2.warpPerspective(bgr, m, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE,
                                    flags=cv2.INTER_CUBIC)
        ratio = w / float(h)
        if math.ceil(48 * ratio) > 160:
            inp = cv2.resize(plate, (160, 48))
        else:
            rw = max(math.ceil(48 * ratio), 48)
            inp = np.zeros((48, 160, 3), np.uint8)
            inp[:, :rw] = cv2.resize(plate, (rw, 48))
        o = self.rt.run(inp.astype(np.float32).reshape(1, 48, 160, 3))
        return self._ctc(o.reshape(20, -1))

    def _ctc(self, probs: np.ndarray):
        """输出已过 softmax（复刻 HyperLPR 上游用法），blank=0"""
        idx = probs.argmax(1)
        confs = probs.max(1)
        out, prev, cs = [], -1, []
        for i, c in enumerate(idx):
            c = int(c)
            if c != prev and c != 0 and c < len(self.chars):
                out.append(self.chars[c])
                cs.append(float(confs[i]))
            prev = c
        return ''.join(out), (float(np.mean(cs)) if cs else 0.0)

    def close(self):
        self.rt.close()


class LprnetReader:
    """rknn_model_zoo LPRNet：94x24 原尺寸 → CTC(blank=67)"""
    def __init__(self, model_path: str, lib_path: str = "librknnrt.so"):
        self.rt = RKNNRuntime(model_path, lib_path)
        self.chars = LPR_CHARS
        self.blank = len(LPR_CHARS) - 1

    def read(self, bgr: np.ndarray, kpt4: np.ndarray):
        M = cv2.getPerspectiveTransform(kpt4.astype(np.float32), DST)
        plate = cv2.warpPerspective(bgr, M, (94, 24))
        o = self.rt.run(plate.astype(np.float32).reshape(1, 24, 94, 3))
        return self._ctc(o.reshape(len(self.chars), 18))

    def _ctc(self, logits: np.ndarray):
        p = np.exp(logits - logits.max(0, keepdims=True))
        p /= p.sum(0, keepdims=True)
        idx = p.argmax(0)
        out, prev, cs = [], -1, []
        for i, c in enumerate(idx):
            c = int(c)
            if c != prev and c != self.blank:
                out.append(self.chars[c])
                cs.append(float(p[c, i]))
            prev = c
        return ''.join(out), (float(np.mean(cs)) if cs else 0.0)

    def close(self):
        self.rt.close()


def make_reader(model_path: str, lib_path: str = "librknnrt.so"):
    """按文件名自动选择引擎"""
    name = model_path.lower()
    if 'rpv3' in name:
        return Rpv3Reader(model_path, lib_path)
    return LprnetReader(model_path, lib_path)
