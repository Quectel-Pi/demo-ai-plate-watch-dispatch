#!/usr/bin/env python3
"""车牌真值裁决：最佳帧高清裁剪 → 双系统共识。

对每段视频：用 y5fu_640 检测（适配大分辨率），按 det 置信度选最佳帧，
全分辨率裁剪车牌区域（带边距），保存高清 crop，并跑：
  1) HyperLPR3 catcher（640 检测 + rpv3 识别）对全帧
  2) 我们的 LPRNet 对透视校正 crop（2x 放大）
输出共识表 + 存 crop 图供人工核对省份字。
"""
import cv2
import numpy as np
import onnxruntime as ort
import sys

sys.path.insert(0, '/home/q/project/Demo/AI-Plate-Watch/tools/board')
from hlpr_pipeline_test import BLANK, CHARS, DST, ctc_greedy, letter_box

DET640 = '/home/q/.hyperlpr3/20230229/onnx/y5fu_640x_sim.onnx'
LPR = '/home/q/project/Demo/AI-Plate-Watch/models/lprnet_fixed.onnx'
OUT_DIR = '/home/q/work/temp/tmp/codex-Middle/rk3576-plate-demo/crops'


def letter_box(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.zeros((size, size, 3), np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return canvas[:, :, ::-1], r, left, top


def main(video, tag):
    import hyperlpr3 as lpr
    catcher = lpr.LicensePlateCatcher()
    det = ort.InferenceSession(DET640, providers=['CPUExecutionProvider'])
    lpr_net = ort.InferenceSession(LPR, providers=['CPUExecutionProvider'])
    v = cv2.VideoCapture(video)
    fps = v.get(cv2.CAP_PROP_FPS) or 30
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    # 1) 扫描：记录 (帧号, det分数, 角点, 尺寸)
    recs = []
    for i in range(0, n, max(1, int(fps / 2))):
        v.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = v.read()
        if not ok:
            continue
        canvas, r, left, top = letter_box(img, 640)
        x = canvas.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0
        out = det.run(None, {det.get_inputs()[0].name: x})[0].reshape(-1, 15)
        for row in out[out[:, 4] > 0.35]:
            xywh = row[:4]
            wbox, hbox = xywh[2] * r, xywh[3] * r
            recs.append((i, float(row[4]), row[5:13], wbox * hbox, img.copy()))
    print(f'[{tag}] 高分检测 {len(recs)} 条')
    # 2) 按帧聚簇（同一帧只留最高分），再按空间分两条轨迹（简化：按帧号排序分段）
    recs.sort(key=lambda t: (-t[3], -t[1]))
    picked, used_frames = [], set()
    for i, conf, kpt, area, img in recs:
        if i in used_frames:
            continue
        used_frames.add(i)
        picked.append((i, conf, kpt, area, img))
        if len(picked) >= 6:
            break
    # 3) 对每个最佳帧：crop + 双系统识别
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    for rank, (i, conf, kpt, area, img) in enumerate(picked):
        kp = kpt.reshape(4, 2)
        kp_orig = (kp - [left, top]) / r
        # 高清 crop（角点外扩 25% 边距）
        c0 = kp_orig.mean(0)
        rad = np.abs(kp_orig - c0).max(0) * 1.6 + 12
        x0, y0 = np.clip(c0 - rad, 0, [img.shape[1], img.shape[0]]).astype(int)
        x1, y1 = np.clip(c0 + rad, 0, [img.shape[1], img.shape[0]]).astype(int)
        crop = img[y0:y1, x0:x1]
        # 我们的 LPRNet：透视校正；2x 分支 = 原图放大后再 warp（提升采样质量）
        M = cv2.getPerspectiveTransform(kp_orig.astype(np.float32), DST)
        plate = cv2.warpPerspective(img, M, (94, 24))
        M2 = cv2.getPerspectiveTransform((kp_orig * 2).astype(np.float32), DST)
        plate2x = cv2.warpPerspective(cv2.resize(img, None, fx=2, fy=2), M2, (94, 24))
        results = {}
        for scale_name, pl in [('1x', plate), ('2x', plate2x)]:
            px = (pl.astype(np.float32)[None].transpose(0, 3, 1, 2) - 127.5) / 127.5
            o = lpr_net.run(None, {lpr_net.get_inputs()[0].name: px})[0]
            results[scale_name] = ctc_greedy(o.reshape(68, 18))
        # HyperLPR3 对同帧全图（其内部 640+rpv3）
        hl = [(c, round(cf, 2)) for c, cf, tp, bx in catcher(img)]
        # HyperLPR3 对高清 crop
        hl_crop = [(c, round(cf, 2)) for c, cf, tp, bx in catcher(crop)]
        cv2.imwrite(f'{OUT_DIR}/{tag}_f{i:04d}_rank{rank}.jpg', crop)
        print(f'  帧{i} conf={conf:.2f} area={area:.0f} | 我们1x={results["1x"]} 2x={results["2x"]} | HLPR全帧={hl} | HLPR crop={hl_crop}')
    v.release()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
