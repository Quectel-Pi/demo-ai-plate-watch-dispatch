#!/usr/bin/env python3
"""M2 视频流管道主入口（无界面板端测试版）。

用法:
  python3 main.py <video> [--det m.rknn] [--lpr m.rknn] [--realtime]
                   [--step N] [--out DIR] [--save-video]

模式:
  --realtime  按视频原速播放，跟不上时自动跳帧（演示模式）
  默认        逐帧分析模式：每帧都处理，全速
"""
import argparse
import os
import time

import cv2
import numpy as np

from detector import PlateDetector
from reader import DST, make_reader
from session import Session
from tracker import Tracker

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
DATA_DIR = os.environ.get('PLATE_WATCH_DATA',
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'data'))
FONT = None  # 中文绘制在宿主机调试时用 PIL，板上 headless 用英文


def draw_detections(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def run(args):
    det = PlateDetector(args.det, size=640)
    lpr = make_reader(args.lpr)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sess = Session(DATA_DIR, os.path.basename(args.video))
    tracker = Tracker()
    video_name = os.path.basename(args.video)

    def read_fn(kpt8):
        text, conf = lpr.read(frame, kpt8)
        M = cv2.getPerspectiveTransform(kpt8.astype(np.float32), DST)
        plate = cv2.warpPerspective(frame, M, (94, 24))
        return text, conf, plate

    writer = None
    if args.save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(sess.base, 'annotated.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))

    # 跳帧策略：--step>0 固定步进；realtime 模式自适应；默认逐帧
    fixed_step = args.step if args.step > 0 else 0
    step = fixed_step or 1
    n_proc = n_skipped = 0
    t_det = t_lpr = 0.0
    ema_proc = None                      # 处理耗时指数滑动平均
    frame_idx = -1
    next_proc_idx = 0
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        frame_idx += 1
        if not ok:
            break
        if frame_idx < next_proc_idx:
            n_skipped += 1
            continue

        t0 = time.perf_counter()
        dets = det.detect(frame)
        t_det += time.perf_counter() - t0

        t1 = time.perf_counter()
        video_t = frame_idx / fps
        events, closed = tracker.step(frame_idx, video_t, dets, read_fn)
        t_lpr += time.perf_counter() - t1

        proc = time.perf_counter() - t0
        ema_proc = proc if ema_proc is None else 0.8 * ema_proc + 0.2 * proc
        n_proc += 1

        # 自适应步进：跟不上原速则降采样（仅 realtime 且未固定 step）
        if args.realtime and not fixed_step and ema_proc:
            step = max(1, int(ema_proc * fps + 0.999))
        next_proc_idx = frame_idx + (step if (fixed_step or args.realtime) else 1)

        for ev in events:
            sess.add_event(ev)
            print(f"  [确认] {ev['plate']} ({ev['plate_type']}) "
                  f"首次出现 {ev['first_seen_s']}s 确认于 {ev['confirmed_s']}s")
        for tr in closed:
            if not tr.confirmed and tr.total_reads + tr.low_quality_reads >= 2:
                sess.add_unresolved(tr)

        if writer:
            writer.write(draw_detections(frame, dets))

    wall = time.perf_counter() - wall_start
    cap.release()
    if writer:
        writer.release()
    det.close()
    lpr.close()

    json_path, csv_path = sess.export()
    print(f"\n视频: {video_name}  处理帧: {n_proc}/{total} (跳过 {n_skipped})")
    print(f"det 平均: {t_det/n_proc*1000:.1f}ms  det+lpr 平均: "
          f"{(t_det+t_lpr)/n_proc*1000:.1f}ms  分析帧率: {n_proc/wall:.1f}FPS "
          f"(墙钟 {wall:.1f}s / 视频 {total/fps:.1f}s)")
    print(sess.summary())
    print(f"结果: {json_path}\n截图目录: {sess.crop_dir}")
    if args.save_video:
        print(f"标注视频: {os.path.join(sess.base, 'annotated.mp4')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--det', default=os.path.join(MODEL_DIR, 'hlpr_det640_fp16.rknn'))
    ap.add_argument('--lpr', default=os.path.join(MODEL_DIR, 'rpv3_fp16_rk3576.rknn'))
    ap.add_argument('--realtime', action='store_true')
    ap.add_argument('--step', type=int, default=0, help='>0: 每 N 帧处理 1 帧')
    ap.add_argument('--save-video', action='store_true')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
