#!/usr/bin/env python3
"""会话与事件记录：确认事件收集、最佳截图落盘、JSON/CSV 导出。

每次「重新开始」创建新会话（对齐方案第 5 节）。
"""
import csv
import json
import os
import time


class Session:
    def __init__(self, data_dir: str, video_name: str = ''):
        self.id = time.strftime('%Y%m%d_%H%M%S')
        self.video = video_name
        self.started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        self.events = []          # 确认事件
        self.unresolved = []      # 超时未确认轨迹（诊断用）
        self.base = os.path.join(data_dir, self.id)
        self.crop_dir = os.path.join(self.base, 'crops')
        os.makedirs(self.crop_dir, exist_ok=True)

    # ---- 收集确认事件（截图落盘）----
    def add_event(self, ev: dict):
        ev.pop('_pix', None)          # UI 专用 QPixmap/numpy 载荷，不落盘
        crop = ev.pop('best_plate', None)
        fn = f"{ev['plate']}_{ev['track_id']:03d}.jpg"
        if crop is not None:
            import cv2
            # 截图放大 4 倍便于人工核对
            cv2.imwrite(os.path.join(self.crop_dir, fn),
                        cv2.resize(crop, (376, 96), interpolation=cv2.INTER_CUBIC))
        self.events.append(ev)
        return ev

    def add_unresolved(self, tr):
        self.unresolved.append({
            'track_id': tr.tid,
            'first_seen_s': round(tr.first_t, 2),
            'last_seen_s': round(tr.last_t, 2),
            'reads': {k: v[0] for k, v in tr.reads.items()},
            'low_quality_frames': tr.low_quality_reads,
            'short_reads': tr.short_reads,
        })

    # ---- 导出 ----
    def export(self):
        meta = {
            'session_id': self.id,
            'video': self.video,
            'started_at': self.started_at,
            'events': self.events,
            'unresolved': self.unresolved,
        }
        json_path = os.path.join(self.base, 'events.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        csv_path = os.path.join(self.base, 'events.csv')
        if self.events:
            keys = [k for k in self.events[0] if k != 'best_plate']
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
                w.writeheader()
                w.writerows(self.events)
        return json_path, csv_path

    def summary(self) -> str:
        plates = ', '.join(e['plate'] for e in self.events) or '（无）'
        return (f"会话 {self.id}: 确认 {len(self.events)} 条 [{plates}]，"
                f"未确认 {len(self.unresolved)} 条")
