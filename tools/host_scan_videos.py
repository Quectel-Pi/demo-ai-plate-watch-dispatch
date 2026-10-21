#!/usr/bin/env python3
"""宿主机 v2 全链路视频扫描：y5fu 检测 -> 透视校正 -> LPRNet 识别 (onnxruntime)。

用法: python3 host_scan_videos.py <video> [<video> ...] [--step N]
输出每视频的车牌时间轨迹，用于真值标注与 M2 基线。
"""
import argparse
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, '/home/q/project/Demo/AI-Plate-Watch/tools/board')
from hlpr_pipeline_test import (CHARS, BLANK, DST, ctc_greedy, letter_box,
                                decode_y5)

DET_ONNX = '/home/q/project/Demo/AI-Plate-Watch/models/hlpr_det320.onnx'
LPR_ONNX = '/home/q/project/Demo/AI-Plate-Watch/models/lprnet_fixed.onnx'


def scan(path, det, lpr, step=15):
    v = cv2.VideoCapture(path)
    fps = v.get(cv2.CAP_PROP_FPS) or 30
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks = {}
    for i in range(0, n, step):
        v.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = v.read()
        if not ok:
            continue
        canvas, r, left, top = letter_box(img)
        # 原始 ONNX: NCHW 0-1；板端 RKNN: NHWC 0-255（转换时已固化 std=255）
        x = canvas.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0
        out = det.run(None, {det.get_inputs()[0].name: x})[0]
        dets = decode_y5(out, r, left, top)
        for d in dets:
            kp = d[5:13].reshape(4, 2).astype(np.float32)
            M = cv2.getPerspectiveTransform(kp, DST)
            plate = cv2.warpPerspective(img, M, (94, 24))
            # 原始 ONNX: NCHW + (x-127.5)/127.5；板端 RKNN 已固化该归一化
            px = (plate.astype(np.float32)[None].transpose(0, 3, 1, 2) - 127.5) / 127.5
            o = lpr.run(None, {lpr.get_inputs()[0].name: px})[0]
            text = ctc_greedy(o.reshape(68, 18))
            if text:
                tracks.setdefault(text, []).append(i)
    v.release()
    return fps, n, tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+')
    ap.add_argument('--step', type=int, default=15)
    a = ap.parse_args()
    det = ort.InferenceSession(DET_ONNX, providers=['CPUExecutionProvider'])
    lpr = ort.InferenceSession(LPR_ONNX, providers=['CPUExecutionProvider'])
    print(f'输入名: det={det.get_inputs()[0].name}, lpr={lpr.get_inputs()[0].name}')
    for p in a.videos:
        fps, n, tracks = scan(p, det, lpr, a.step)
        print(f'=== {p} ({n}帧 {fps:.0f}fps)')
        for t, frames in sorted(tracks.items(), key=lambda x: -len(x[1])):
            print(f'  {t}: {len(frames)}次  {frames[0]/fps:.1f}s~{frames[-1]/fps:.1f}s')


if __name__ == '__main__':
    main()
