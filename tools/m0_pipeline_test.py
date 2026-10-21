#!/usr/bin/env python3
"""M0 宿主机全链路验证：HF 四角点检测 → 透视校正 → LPRNet ONNX 识别。

用法: python3 tools/m0_pipeline_test.py <img_or_dir> [--save_dir DIR]
输出: 每张图的检测框/角点标注图 + 校正车牌图 + 识别文本。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# 角点顺序（HF 权重约定）: 0:TL 1:TR 2:BR 3:BL
DST = np.float32([[0, 0], [93, 0], [93, 23], [0, 23]])  # 94x24

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']


def ctc_greedy(preds):
    """preds: [68, T] -> 文本"""
    seq = np.argmax(preds, axis=0)
    out, prev = [], None
    for c in seq:
        c = int(c)
        if c != prev and c != len(CHARS) - 1:
            out.append(CHARS[c])
        prev = c
    return ''.join(out)


def run_pipeline(detector, lpr, img_path, save_dir=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    r = detector.predict(img, conf=0.25, save=False, verbose=False)[0]
    results = []
    for box, kpts in zip(r.boxes, r.keypoints):
        pts = kpts.xy.cpu().numpy().reshape(-1, 2).astype(np.float32)  # [4,2] TL TR BR BL
        # 透视校正
        M = cv2.getPerspectiveTransform(pts, DST)
        plate = cv2.warpPerspective(img, M, (94, 24))
        # LPRNet: BGR, 归一化 [-1,1]（RKNN 转换时 mean/std=127.5 固化进模型，
        # 板上直传 0-255；宿主机 ONNX 需手动归一化）
        x = (plate.astype(np.float32) - 127.5) / 127.5
        x = x.transpose(2, 0, 1).reshape(1, 3, 24, 94)
        out = lpr.run(None, {'input': x})[0][0]  # [68,18]
        text = ctc_greedy(out)
        conf = float(box.conf[0])
        results.append((text, conf, pts, plate))
        print(f"  [{img_path.name}] conf={conf:.2f} 车牌={text}")
    if save_dir and results:
        save_dir.mkdir(parents=True, exist_ok=True)
        vis = img.copy()
        for text, conf, pts, plate in results:
            p = pts.astype(int)
            cv2.polylines(vis, [p], True, (0, 255, 0), 2)
            # cv2.putText 不支持中文：图上画 ASCII 化文本，真实车牌在文件名保留
            ascii_text = text.encode('ascii', 'replace').decode()
            cv2.putText(vis, ascii_text, (int(p[0][0]), int(p[0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(save_dir / f"{img_path.stem}_vis.jpg"), vis)
        for i, (text, conf, pts, plate) in enumerate(results):
            cv2.imwrite(str(save_dir / f"{img_path.stem}_plate{i}_{text}.jpg"), plate)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--det", default="downloads/plate_keypoint.pt")
    ap.add_argument("--lpr", default="models/lprnet_fixed.onnx")
    ap.add_argument("--save_dir", default="assets/m0_out")
    args = ap.parse_args()

    detector = YOLO(args.det)
    lpr = ort.InferenceSession(args.lpr, providers=['CPUExecutionProvider'])

    src = Path(args.src)
    imgs = sorted(src.glob("*.jpg")) if src.is_dir() else [src]
    if not imgs:
        sys.exit(f"no images in {src}")
    save = Path(args.save_dir)
    for p in imgs:
        run_pipeline(detector, lpr, p, save)
    print(f"[done] 标注与车牌图已存 {save}/")


if __name__ == "__main__":
    main()
