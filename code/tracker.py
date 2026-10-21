#!/usr/bin/env python3
"""跟踪与多帧确认：位置关联（IoU）+ 读数累计表决 + 最佳截图保留。

规则（对齐方案第 5 节）：
- 同一车牌连续出现归入同一条轨迹（IoU 关联）
- 每条轨迹只确认一次事件；轨迹消失超时后归档
- 确认条件：同一读数出现 >= confirm_min 次且占比 >= confirm_ratio，
  且该读数平均置信度 >= confirm_min_conf（防止远距离低置信垃圾读数凑票确认）
- 长度门闸：读数 < 7 字符不参与表决（完整蓝牌 7 位/绿牌 8 位；
  半可见/刚进画面车牌会高置信读出「苏A」这类截断前缀，必须等完整读数）
- 质量门闸：车牌像素高 < min_h_px 的帧不参与表决（记为"待确认"）
- 最佳截图：det置信度 × sqrt(车牌面积) 最高的一帧
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    tid: int
    box: np.ndarray                    # 最近一帧 [x1,y1,x2,y2]
    first_frame: int
    first_t: float
    last_frame: int
    last_t: float
    lost_frames: int = 0
    reads: dict = field(default_factory=dict)   # text -> [count, conf_sum]
    total_reads: int = 0
    low_quality_reads: int = 0
    short_reads: int = 0                # <7 字符的截断读数（不参与表决）
    best_q: float = -1.0                # 最佳帧质量分
    best_plate: np.ndarray = None       # 最佳帧校正图（94x24x3）
    best_frame: int = -1
    confirmed: bool = False
    confirmed_text: str = ''
    confirmed_t: float = 0.0

    def quality(self, det_conf: float, kpt8: np.ndarray):
        h = np.linalg.norm(kpt8[2] - kpt8[1])   # 右侧上下角点距离 ≈ 牌高
        return det_conf * np.sqrt(max(h, 1.0)), h


class Tracker:
    def __init__(self, iou_thresh=0.2, max_lost_s=2.5, confirm_min=3,
                 confirm_ratio=0.5, min_h_px=14.0, confirm_min_conf=0.80):
        self.iou_thresh = iou_thresh
        self.max_lost_s = max_lost_s
        self.confirm_min = confirm_min
        self.confirm_ratio = confirm_ratio
        self.min_h_px = min_h_px
        self.confirm_min_conf = confirm_min_conf
        self.tracks: list[Track] = []
        self._next_tid = 1

    # ---- 主步进：det=[n,14] 检出，read_fn(bgr,kpt)->(text,conf,plate) ----
    # 返回 (new_events, closed_tracks)
    def step(self, frame_idx: int, t_s: float, dets: np.ndarray, read_fn):
        new_events, closed = [], []
        matched_tracks = set()

        # 1) 匹配：贪心 IoU
        for d in dets:
            best_iou, best_t = self.iou_thresh, None
            for tr in self.tracks:
                if tr.tid in matched_tracks:
                    continue
                iou = self._iou(tr.box, d[:4])
                if iou >= best_iou:
                    best_iou, best_t = iou, tr
            if best_t is None:
                tr = Track(tid=self._next_tid, box=d[:4].copy(),
                           first_frame=frame_idx, first_t=t_s,
                           last_frame=frame_idx, last_t=t_s)
                self._next_tid += 1
                self.tracks.append(tr)
            else:
                tr = best_t
            matched_tracks.add(tr.tid)
            tr.box = d[:4].copy()
            tr.last_frame, tr.last_t, tr.lost_frames = frame_idx, t_s, 0
            if not tr.confirmed:   # 已确认轨迹不再消耗 NPU 识别（事件已冻结）
                self._update_read(tr, d, frame_idx, t_s, read_fn)
                ev = self._maybe_confirm(tr, t_s)
                if ev:
                    new_events.append(ev)

        # 2) 未匹配轨迹：超时归档（含未确认轨迹，供诊断）
        alive = []
        for tr in self.tracks:
            if tr.tid in matched_tracks or t_s - tr.last_t <= self.max_lost_s:
                alive.append(tr)
            else:
                closed.append(tr)
        self.tracks = alive
        return new_events, closed

    # ---- 当前活跃轨迹快照（UI 三状态展示用）----
    def snapshot(self):
        """返回每条活跃轨迹的实时状态：detected(检测中)/candidate(候选中)/confirmed(已确认)"""
        out = []
        for tr in self.tracks:
            if tr.confirmed:
                out.append({
                    'tid': tr.tid, 'box': tr.box.copy(), 'state': 'confirmed',
                    'text': tr.confirmed_text, 'votes': 0, 'need': self.confirm_min,
                    'conf': 0.0,
                    'plate_type': 'green' if len(tr.confirmed_text) >= 8 else 'blue',
                    'best_plate': tr.best_plate,
                })
                continue
            if tr.total_reads == 0:
                out.append({
                    'tid': tr.tid, 'box': tr.box.copy(), 'state': 'detected',
                    'text': '', 'votes': 0, 'need': self.confirm_min, 'conf': 0.0,
                    'plate_type': '', 'best_plate': None,
                })
                continue
            text, (cnt, conf_sum) = max(tr.reads.items(), key=lambda kv: kv[1][0])
            out.append({
                'tid': tr.tid, 'box': tr.box.copy(), 'state': 'candidate',
                'text': text, 'votes': cnt, 'need': self.confirm_min,
                'conf': conf_sum / max(1, cnt),
                'plate_type': 'green' if len(text) >= 8 else 'blue',
                'best_plate': tr.best_plate,
            })
        return out

    # ---- 轨迹内读数累计 ----
    def _update_read(self, tr: Track, det: np.ndarray, frame_idx: int,
                     t_s: float, read_fn):
        det_conf = float(det[4])
        kpt8 = det[5:13].reshape(4, 2)
        q, h_px = tr.quality(det_conf, kpt8)
        is_best = q > tr.best_q
        if is_best:
            tr.best_q, tr.best_frame = q, frame_idx
        if h_px < self.min_h_px:
            tr.low_quality_reads += 1     # 过小车牌：不参与表决（待确认）
            return
        text, conf, plate = read_fn(kpt8)
        if not text:
            return
        if len(text) < 7:      # 截断读数（半可见/遮挡）：高置信但非完整牌号，不参与表决
            tr.short_reads += 1
            return
        if is_best:
            tr.best_plate = plate.copy()
        tr.reads.setdefault(text, [0, 0.0])
        tr.reads[text][0] += 1
        tr.reads[text][1] += conf
        tr.total_reads += 1

    # ---- 多帧表决确认 ----
    def _maybe_confirm(self, tr: Track, t_s: float):
        if tr.confirmed or tr.total_reads < self.confirm_min:
            return None
        text, (cnt, conf_sum) = max(tr.reads.items(), key=lambda kv: kv[1][0])
        avg_conf = conf_sum / max(1, cnt)
        if (cnt >= self.confirm_min and cnt / tr.total_reads >= self.confirm_ratio
                and avg_conf >= self.confirm_min_conf):
            tr.confirmed = True
            tr.confirmed_text = text
            tr.confirmed_t = t_s
            return self._to_event(tr)

    def _to_event(self, tr: Track):
        avg_conf = tr.reads[tr.confirmed_text][1] / max(1, tr.reads[tr.confirmed_text][0])
        return {
            'track_id': tr.tid,
            'plate': tr.confirmed_text,
            'plate_type': 'green' if len(tr.confirmed_text) >= 8 else 'blue',
            'first_seen_s': round(tr.first_t, 2),
            'confirmed_s': round(tr.confirmed_t, 2),
            'frames': tr.total_reads,
            'low_quality_frames': tr.low_quality_reads,
            'char_conf': round(avg_conf, 3),
            'best_q': round(tr.best_q, 2),
            'best_plate': tr.best_plate,
        }

    @staticmethod
    def _iou(a, b):
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
