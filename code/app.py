#!/usr/bin/env python3
"""智能车牌识别调度系统 · 网约车接客调度演示终端（PySide6 深色科技风 / 中英双语）。

布局：顶栏品牌+识别状态+调度统计 / 左侧车辆识别（视频=进站模拟、摄像头=实时）/
      右侧上客区（大字播报横幅 + A01~A04 车位卡）/ 下方候车队列（登记/分配/操作）。

视频与摄像头共用同一条进站触发管线：识别确认事件 → DispatchCore 调度
（登记分配、FIFO 补位、超时巡检、未登记补登记）。
推理全部在 QThread 内完成（NPU + OpenCV），UI 线程只负责渲染，
视频帧上的中文标注用 QPainter 叠加（cv2.putText 不支持中文）。
摄像头模式：V4L2 QUERYCAP 过滤单平面捕获节点（排除 rkisp mplane 节点），
USB 总线优先排序，open+read 实测出帧后启用；BUFFERSIZE=1 保证低延迟。
"""
import fcntl
import glob
import json
import os
import re
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import (QImage, QPixmap, QPainter, QColor, QFont, QPen,
                           QBrush, QPalette, QFontDatabase)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QMessageBox, QSizePolicy,
    QInputDialog, QGridLayout,
)

from detector import PlateDetector
from reader import DST, make_reader
from session import Session
from tracker import Tracker
from dispatch import (DispatchCore, PRESET_VEHICLES, DEMO_SCHEDULE,
                      ORDER_WAITING, ORDER_ASSIGNED, ORDER_SERVING)
from voice_player import VoicePlayer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..'))
MODEL_DIR = os.path.join(APP_ROOT, 'models')
DATA_DIR = os.environ.get('PLATE_REC_DATA',
                          os.environ.get('PLATE_WATCH_DATA',
                                         os.path.join(APP_ROOT, 'data')))
VIDEO_DIR = os.path.join(APP_ROOT, 'materials', 'videos')

# ================= i18n：中英双语 =================
LANG = 'en' if os.environ.get('PLATE_LANG', 'zh').lower().startswith('en') else 'zh'

TEXTS = {
    'zh': {
        'app_title': '智能车牌识别调度系统',
        'meta_idle': '未加载视频源',
        'stat_ready': '就绪',
        'stat_finished': '识别结束',
        'stat_wait_end': '会话结束后可导出',
        'btn_open': '打开视频',
        'btn_camera': '摄像头识别',
        'btn_pause': '暂停',
        'btn_resume': '继续',
        'btn_restart': '重新会话',
        'btn_export': '导出记录',
        'cand_title': '候选中车牌 · 多帧确认',
        'cand_empty': '等待车牌进入画面…',
        'cand_row': '{text}    {votes}/{need} 票    置信 {conf:.2f}',
        'ev_title': '已确认通行事件',
        'count_fmt': '确认 {confirmed} · 候选 {candidates}',
        'meta_video': '{name}  |  {fps:.1f} FPS · 检测 {ms:.1f}ms · 帧 {frame}/{total} · {t:.1f}s',
        'meta_camera': '{name}  |  {fps:.1f} FPS · 检测 {ms:.1f}ms · 实时 {t:.1f}s',
        'placeholder': '打开视频或启用摄像头开始识别',
        'state_detected': '检测中',
        'state_candidate': '{text}  {votes}/{need}票',
        'plate_blue': '蓝牌',
        'plate_green': '新能源绿牌',
        'card_sub': '{ptype} · 置信 {conf:.2f}\n{t:05.2f}s 确认 · #{tid}',
        # ---- 调度终端 ----
        'banner_idle': '等待识别已登记车辆 …',
        'banner_dup': '{plate} 已有进行中订单，重复登记被拒',
        'banner_demo': '预演模式：自动演示到位/完成/超时/取消/补登记/二次接客全流程',
        'banner_assign': '{plate}，请前往 {spot}',
        'banner_wait': '{plate}，车位已满，请在候车队列等待',
        'banner_unreg': '发现未登记车辆 {plate}，可在队列中补登记',
        'banner_timeout': '{plate} 已超时未到位（{spot}），可取消分配',
        # ---- 历史条短语（横幅下小字，最近 3 条） ----
        'hist_registered': '{plate} 已登记',
        'hist_assigned': '{plate} → {spot}',
        'hist_reannounce': '{plate} → {spot} 重播',
        'hist_arrived': '{plate} 到位 {spot}',
        'hist_completed': '{plate} 完成',
        'hist_cancelled': '{plate} 取消',
        'hist_timeout': '{plate} 超时',
        'hist_unregistered': '{plate} 未登记',
        'hist_still_waiting': '{plate} 等待',
        'hist_register_rejected': '{plate} 重复登记被拒',
        'banner_arrived': '{plate} 已到位 {spot}，开始接客',
        'banner_done': '{plate} 接客完成，{spot} 已释放',
        'banner_cancel': '{plate} 订单已取消',
        'zone_pickup': '上客区 · 车位状态',
        'zone_queue': '候车队列',
        'queue_empty': '暂无候车订单 · 点击「登记乘客用车」或「预置演示车辆」',
        'spot_free': '空闲',
        'spot_assigned': '已分配',
        'spot_serving': '接客中',
        'ost_wait': '等待车位',
        'ost_timeout': '已超时',
        'unreg_row': '未登记',
        'stat_dispatch': '完成 {done} · 等待 {waiting} · 车位 {used}/{total}',
        'btn_register': '登记乘客用车',
        'btn_preset': '预置演示车辆',
        'btn_demo': '预演模式',
        'btn_arrive': '到位',
        'btn_finish': '接客完成',
        'btn_cancel': '取消',
        'btn_regen': '补登记',
        'dlg_reg_title': '登记乘客用车',
        'dlg_reg_hint': '输入车牌（7~8 位，如 皖ADJ6382）',
        'dlg_reg_bad': '车牌格式不正确（需 7~8 位）',
        'dlg_reg_dup': '{plate} 已有进行中的订单',
        'log_loading_det': '加载检测模型 …',
        'log_loading_lpr': '加载识别模型 …',
        'log_models_ready': '模型就绪',
        'log_cannot_open': '打不开视频源: {path}',
        'log_start_video': '开始分析: {name}',
        'log_search_cam': '正在搜索摄像头 …',
        'log_cam_none': '未找到可用摄像头',
        'log_cam_open': '摄像头已打开: {dev} ({w}×{h})',
        'log_cam_lost': '摄像头读取失败，会话结束',
        'log_start_cam': '开始实时识别',
        'log_session_done': '会话结束，确认 {n} 条 · 记录: {path}',
        'log_exported': '已导出当前会话: {path}',
        'dlg_pick_video': '选择车辆视频',
        'dlg_cam_title': '摄像头识别',
        'dlg_no_cam': '未找到可用摄像头。\n请接入 USB/CSI 摄像头后重试。',
        'dlg_export': '导出记录',
        'dlg_exported': '记录已导出到:\n{path}',
        'btn_ok': '知道了',
    },
    'en': {
        'app_title': 'Plate Recognition & Dispatch System',
        'meta_idle': 'No source loaded',
        'stat_ready': 'Ready',
        'stat_finished': 'Recognition ended',
        'stat_wait_end': 'Export available after the session ends',
        'btn_open': 'Open Video',
        'btn_camera': 'Live Camera',
        'btn_pause': 'Pause',
        'btn_resume': 'Resume',
        'btn_restart': 'New Session',
        'btn_export': 'Export',
        'cand_title': 'Candidates · Multi-frame Voting',
        'cand_empty': 'Waiting for a plate to appear…',
        'cand_row': '{text}    {votes}/{need} votes    conf {conf:.2f}',
        'ev_title': 'Confirmed Passages',
        'count_fmt': 'Confirmed {confirmed} · Candidates {candidates}',
        'meta_video': '{name}  |  {fps:.1f} FPS · det {ms:.1f}ms · frame {frame}/{total} · {t:.1f}s',
        'meta_camera': '{name}  |  {fps:.1f} FPS · det {ms:.1f}ms · live {t:.1f}s',
        'placeholder': 'Open a video or enable the camera to start',
        'state_detected': 'Detecting',
        'state_candidate': '{text}  {votes}/{need}',
        'plate_blue': 'Blue plate',
        'plate_green': 'Green plate',
        'card_sub': '{ptype} · conf {conf:.2f}\nconfirmed at {t:05.2f}s · #{tid}',
        # ---- dispatch terminal ----
        'banner_idle': 'Waiting for registered vehicles …',
        'banner_dup': '{plate} already has an active order — duplicate rejected',
        'banner_demo': 'DEMO mode: auto-driving arrive/finish/timeout/cancel/re-register flows',
        'banner_assign': '{plate}, please go to {spot}',
        'banner_wait': '{plate}, all spots busy — please wait in queue',
        'banner_unreg': 'Unregistered vehicle {plate} — register in queue',
        'banner_timeout': '{plate} has not arrived ({spot}) — may cancel',
        # ---- history strip (last 3 events under banner) ----
        'hist_registered': '{plate} registered',
        'hist_assigned': '{plate} → {spot}',
        'hist_reannounce': '{plate} → {spot} repeat',
        'hist_arrived': '{plate} arrived {spot}',
        'hist_completed': '{plate} done',
        'hist_cancelled': '{plate} cancelled',
        'hist_timeout': '{plate} timeout',
        'hist_unregistered': '{plate} unregistered',
        'hist_still_waiting': '{plate} waiting',
        'hist_register_rejected': '{plate} duplicate rejected',
        'banner_arrived': '{plate} arrived at {spot}, now serving',
        'banner_done': '{plate} pickup finished, {spot} released',
        'banner_cancel': 'Order for {plate} cancelled',
        'zone_pickup': 'PICKUP ZONE',
        'zone_queue': 'WAITING QUEUE',
        'queue_empty': 'No orders · use "Register Ride" or "Preset Fleet"',
        'spot_free': 'Free',
        'spot_assigned': 'Assigned',
        'spot_serving': 'Serving',
        'ost_wait': 'Waiting for spot',
        'ost_timeout': 'Timed out',
        'unreg_row': 'Unregistered',
        'stat_dispatch': 'Done {done} · Waiting {waiting} · Spots {used}/{total}',
        'btn_register': 'Register Ride',
        'btn_preset': 'Preset Fleet',
        'btn_demo': 'Demo Mode',
        'btn_arrive': 'Arrived',
        'btn_finish': 'Finish',
        'btn_cancel': 'Cancel',
        'btn_regen': 'Register',
        'dlg_reg_title': 'Register a Ride',
        'dlg_reg_hint': 'Enter plate (7-8 chars, e.g. 皖ADJ6382)',
        'dlg_reg_bad': 'Invalid plate format (7-8 characters required)',
        'dlg_reg_dup': '{plate} already has an active order',
        'log_loading_det': 'Loading detection model …',
        'log_loading_lpr': 'Loading recognition model …',
        'log_models_ready': 'Models ready',
        'log_cannot_open': 'Cannot open source: {path}',
        'log_start_video': 'Analyzing: {name}',
        'log_search_cam': 'Searching for camera …',
        'log_cam_none': 'No usable camera found',
        'log_cam_open': 'Camera opened: {dev} ({w}×{h})',
        'log_cam_lost': 'Camera read failure, session ended',
        'log_start_cam': 'Live recognition started',
        'log_session_done': 'Session ended, {n} confirmed · log: {path}',
        'log_exported': 'Session exported: {path}',
        'dlg_pick_video': 'Choose a vehicle video',
        'dlg_cam_title': 'Live Camera',
        'dlg_no_cam': 'No usable camera found.\nConnect a USB/CSI camera and retry.',
        'dlg_export': 'Export',
        'dlg_exported': 'Records exported to:\n{path}',
        'btn_ok': 'OK',
    },
}


def tr(key, **kw):
    """按当前语言取文案；缺失时回退中文，再回退 key 本身"""
    d = TEXTS.get(LANG) or TEXTS['zh']
    s = d.get(key)
    if s is None:
        s = TEXTS['zh'].get(key, key)
    return s.format(**kw) if kw else s


def set_lang(lang):
    global LANG
    LANG = 'en' if str(lang).lower().startswith('en') else 'zh'


def resolve_video(p):
    """相对路径依次尝试 cwd、应用根目录（run.sh 会 cd 到 code/）"""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    for base in (os.getcwd(), APP_ROOT):
        cand = os.path.join(base, p)
        if os.path.isfile(cand):
            return cand
    return p


# ---- 深色主题配色（状态色克制使用，参考现代安防终端审美） ----
COL_DETECTED = QColor(148, 163, 184)    # 石板灰：检测中
COL_CANDIDATE = QColor(251, 191, 36)    # 琥珀：候选中
COL_CONFIRMED = QColor(52, 211, 153)    # 翡翠绿：已确认
COL_BLUE = QColor(96, 165, 250)        # 蓝牌号码色
COL_GREEN = QColor(52, 211, 153)        # 绿牌号码色
COL_TAGTEXT = QColor(11, 18, 32)        # 彩色标签块上的深色文字

APP_QSS = """
/* ============ 深色科技风（安防监控大屏 / AI 检测系统）============
   暗底 #0b1220 / 青绿 #22d3ee 高亮 / 琥珀 #fbbf24 候选 /
   等宽字体用于数据读出（FPS·票数·牌号）/ 发光边缘用 QSS 渐变模拟 */
QWidget { background: transparent;
          font-family: 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif; }
QMainWindow { background: #0b1220; }
QToolTip { background: #0e1729; color: #e5e7eb;
           border: 1px solid #2a3b5e; padding: 4px 8px; }
QLabel { color: #e5e7eb; }
QLabel#brand    { color: #f8fafc; font-size: 19px; font-weight: 700;
                  letter-spacing: 1px; }
/* 状态指示点：径向渐变自发光（中心实心 + 外圈光晕） */
QLabel#dot      { border-radius: 5px;
                  background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    stop:0 #22d3ee, stop:0.45 #22d3ee, stop:1 rgba(34,211,238,0)); }
/* 数据读出区一律等宽字体 */
QLabel#meta      { color: #67e8f9; font-size: 12px;
                   font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                                'Noto Sans Mono CJK SC', monospace; }
QLabel#stat      { color: #34d399; font-size: 12px;
                   font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                                'Noto Sans Mono CJK SC', monospace; }
QLabel#count     { color: #cbd5e1; font-size: 14px; font-weight: 700;
                   font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                                'Noto Sans Mono CJK SC', monospace; }
QLabel#panelcap  { color: #fbbf24; font-weight: 700; font-size: 13px;
                   letter-spacing: 2px; }
QLabel#evhead    { color: #e5e7eb; font-weight: 700; font-size: 14px; }
QLabel#candempty { color: #8b95a7; font-size: 13px; }
QLabel#candrow   { color: #fde68a; font-size: 15px; font-weight: 700;
                   font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                                'Noto Sans Mono CJK SC', monospace; }
/* 面板发光边缘：暗青描边 + 顶部亮线（模拟背光溢出） */
QFrame#panel  { background: #0e1729; border-radius: 12px;
                border: 1px solid rgba(56,189,248,55);
                border-top: 1px solid rgba(103,232,249,95); }
QFrame#candbox{ background: #0e1729; border-radius: 12px;
                border: 1px solid rgba(251,191,36,75);
                border-top: 1px solid rgba(253,230,138,110); }
QPushButton { background: #16213a; color: #d7e0ee;
              border: 1px solid #2a3b5e; border-radius: 8px;
              padding: 6px 16px; font-size: 14px; }
QPushButton:hover  { background: #1d2b4a; border-color: rgba(103,232,249,130); }
QPushButton:pressed{ background: #223358; }
QPushButton:disabled { color: #5b6b85; border-color: #1d2b45; background: #101a2e; }
/* 主按钮：青色渐变 + 亮边（发光感），等宽不适用、保持粗体 */
QPushButton#primary {
    color: #032530; font-weight: 700;
    border: 1px solid rgba(148,247,255,140);
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
               stop:0 #2ee6f7, stop:1 #0fb3c7); }
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
               stop:0 #54ecfb, stop:1 #18c6da); }
QPushButton#primary:pressed { background: #12a5b8; }
QPushButton#primary:disabled { background: #0e5560; color: #6fa3ab;
                              border-color: #0e5560; }
QPushButton#lang { background: #122036; color: #7dd3fc;
                   border: 1px solid rgba(103,232,249,110); border-radius: 14px;
                   font-size: 13px; padding: 4px 14px; }
QPushButton#lang:hover { background: #18304f; }
/* ---- 对话框深色统一（QFileDialog/QMessageBox 都是 QDialog）----
   原生 GTK 对话框不吃 QSS，QFileDialog 必须配合 DontUseNativeDialog 使用 */
QDialog { background: #0e1729; }
QMessageBox QLabel { color: #e5e7eb; font-size: 14px; }
QFileDialog QListView, QFileDialog QTreeView {
    background: #0b1220; color: #e5e7eb;
    border: 1px solid #1d2b45; border-radius: 8px;
    selection-background-color: #18c3d8; selection-color: #032530; }
QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
    background: #1d2b4a; }
QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
    background: #18c3d8; color: #032530; }
QFileDialog QLineEdit {
    background: #0b1220; color: #e5e7eb;
    border: 1px solid #2a3b5e; border-radius: 6px; padding: 5px 8px;
    selection-background-color: #18c3d8; selection-color: #032530; }
QFileDialog QComboBox, QComboBox {
    background: #16213a; color: #d7e0ee;
    border: 1px solid #2a3b5e; border-radius: 6px; padding: 5px 10px; }
QComboBox QAbstractItemView {
    background: #0e1729; color: #e5e7eb;
    selection-background-color: #18c3d8; selection-color: #032530; }
QFileDialog QToolBar { background: transparent; border: none; spacing: 4px; }
QFileDialog QToolButton {
    background: transparent; color: #d7e0ee; border: none;
    border-radius: 6px; padding: 6px 10px; }
QFileDialog QToolButton:hover  { background: #1d2b4a; }
QFileDialog QToolButton:pressed { background: #223358; }
QDialogButtonBox QPushButton { min-width: 96px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0b1220; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #263a5c; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
/* ---- 调度终端：大字播报横幅（青色发光底） ---- */
QLabel#banner { color: #032530; font-size: 22px; font-weight: 800;
                border-radius: 10px; padding: 14px 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                           stop:0 #2ee6f7, stop:1 #0fb3c7);
                border: 1px solid rgba(148,247,255,150); }
QLabel#banner[urgent="true"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                           stop:0 #fbbf24, stop:1 #f59e0b); }
/* ---- 最近事件历史条（横幅下小字，同时发生的事件一目了然） ---- */
QLabel#hist { color: #7dd3fc; font-size: 12px; font-weight: 600;
              font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
              padding: 2px 6px; background: rgba(15,25,48,120);
              border-radius: 6px; }
/* ---- 车位卡 ---- */
QFrame#spotcard { background: #0f1930; border: 1px solid #223352;
                  border-radius: 10px; }
QLabel#spotname { color: #67e8f9; font-size: 15px; font-weight: 800;
                  font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; }
QLabel#spotplate { color: #e5e7eb; font-size: 14px; font-weight: 700;
                   font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                                'Noto Sans Mono CJK SC', monospace; }
/* ---- 队列行 ---- */
QFrame#queuecard { background: #0f1930; border: 1px solid #223352;
                   border-radius: 10px; }
QFrame#queuecard[st="waiting"]  { border-left: 3px solid #64748b; }
QFrame#queuecard[st="assigned"] { border-left: 3px solid #fbbf24; }
QFrame#queuecard[st="serving"]  { border-left: 3px solid #34d399; }
QFrame#queuecard[st="timeout"]  { border-left: 3px solid #f87171; }
QFrame#queuecard[st="unreg"]    { border-left: 3px solid #7dd3fc; }
QLabel#qplate { color: #fde68a; font-size: 14px; font-weight: 700;
                font-family: 'DejaVu Sans Mono', 'Liberation Mono',
                             'Noto Sans Mono CJK SC', monospace; }
QLabel#qstate { color: #8b95a7; font-size: 12px; }
QLabel#qempty { color: #8b95a7; font-size: 13px; }
QPushButton#qbtn { background: #16213a; color: #d7e0ee;
                   border: 1px solid #2a3b5e; border-radius: 6px;
                   padding: 4px 12px; font-size: 12px; }
QPushButton#qbtn:hover { background: #1d2b4a; border-color: rgba(103,232,249,130); }
"""


def build_dark_palette():
    """Fusion 风格配套深色调色板：兜住 QSS 覆盖不到的原生控件
    （对话框编辑框光标、下拉弹层、tooltip 等），保证全界面无浅色残留"""
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor('#0b1220'))
    pal.setColor(QPalette.WindowText, QColor('#e5e7eb'))
    pal.setColor(QPalette.Base, QColor('#0a101d'))
    pal.setColor(QPalette.AlternateBase, QColor('#0e1729'))
    pal.setColor(QPalette.Text, QColor('#e5e7eb'))
    pal.setColor(QPalette.Button, QColor('#16213a'))
    pal.setColor(QPalette.ButtonText, QColor('#d7e0ee'))
    pal.setColor(QPalette.ToolTipBase, QColor('#0e1729'))
    pal.setColor(QPalette.ToolTipText, QColor('#e5e7eb'))
    pal.setColor(QPalette.Highlight, QColor('#18c3d8'))
    pal.setColor(QPalette.HighlightedText, QColor('#032530'))
    pal.setColor(QPalette.Link, QColor('#22d3ee'))
    pal.setColor(QPalette.PlaceholderText, QColor('#8b95a7'))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor('#5b6b85'))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#5b6b85'))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#5b6b85'))
    return pal


def bgr_to_qimg(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def _v4l2_probe(path):
    """V4L2 能力查询。返回 (是否单平面捕获设备, bus_info)。
    rkisp 等 mplane 节点没有 0x1 捕获位，自动排除——板上 CSI 管道未配置时
    其 capture 位为真但 STREAMON 永远失败（实测）。"""
    VIDIOC_QUERYCAP = 0x80685600
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        finally:
            os.close(fd)
    except OSError:
        return False, ""
    caps, dev_caps = struct.unpack_from("<II", buf, 84)
    eff = dev_caps if caps & 0x80000000 else caps   # V4L2_CAP_DEVICE_CAPS
    bus = bytes(buf[48:80]).split(b"\0", 1)[0].decode(errors="ignore")
    return bool(eff & 0x1), bus                     # V4L2_CAP_VIDEO_CAPTURE


def find_cameras():
    """返回能实际出图的摄像头索引列表（USB 优先，open+read 实测验证）。
    沿用健身项目验证过的发现策略：QUERYCAP 过滤 + USB 优先 + 实测出帧。"""
    devs = sorted((d for d in glob.glob('/dev/video*')
                   if re.fullmatch(r'/dev/video\d+', d)),
                  key=lambda p: int(p.rsplit('video', 1)[1]))
    cands = []
    for d in devs:
        is_cap, bus = _v4l2_probe(d)
        if is_cap:
            n = int(d.rsplit('video', 1)[1])
            cands.append((0 if bus.startswith('usb') else 1, n))
    cands.sort()
    out = []
    for _, n in cands:
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
        if not cap.open(n, cv2.CAP_V4L2):
            cap.release()
            continue
        ok, _f = cap.read()
        cap.release()
        if ok:
            out.append(n)
    return out


class Worker(QThread):
    """推理线程：解码 → 检测 → 跟踪/识别/确认，信号回传 UI。
    source: {'type':'video','path':...} 或 {'type':'camera','dev':路径|int|None}
    dev=None 表示自动扫描第一个可用摄像头。"""
    frameReady = Signal(object, list, dict)   # QImage, snapshots, stats
    eventConfirmed = Signal(dict)
    logMsg = Signal(str)
    videoFinished = Signal(str, str)          # summary, json_path
    exportDone = Signal(str)                  # 摄像头会话即时导出
    camFailed = Signal()                      # 未找到/打不开摄像头

    def __init__(self):
        super().__init__()
        self.source = None
        self.paused = False
        self._stop = False
        self._restart = False
        self.export_requested = False
        self.det = None
        self.lpr = None

    # ---- 供 UI 调用的控制接口 ----
    def open_video(self, path):
        self.source = {'type': 'video', 'path': path}
        self._queue_start()

    def open_camera(self, dev=None):
        self.source = {'type': 'camera', 'dev': dev}
        self._queue_start()

    def _queue_start(self):
        self._restart = True
        self._stop = False
        if not self.isRunning():
            self.start()

    def toggle_pause(self):
        self.paused = not self.paused

    def restart_session(self):
        self._restart = True
        self.paused = False

    def stop(self):
        self._stop = True

    # ---- 模型懒加载（线程内只加载一次）----
    def _ensure_models(self):
        if self.det is None:
            _t = time.time()
            self.logMsg.emit(tr('log_loading_det'))
            self.det = PlateDetector(os.path.join(MODEL_DIR, 'hlpr_det640_fp16.rknn'),
                                     size=640)
            self.logMsg.emit(tr('log_loading_lpr'))
            self.lpr = make_reader(os.path.join(MODEL_DIR, 'rpv3_fp16_rk3576.rknn'))
            print(f"[init] models loaded in {time.time()-_t:.2f}s", flush=True)
            self.logMsg.emit(tr('log_models_ready'))

    def run(self):
        self._ensure_models()
        while not self._stop:
            if self._restart:
                self._restart = False
                self._play_one()
            else:
                self.msleep(80)
        if self.det:
            self.det.close()
        if self.lpr:
            self.lpr.close()

    def _play_one(self):
        src = self.source
        if not src:
            return
        if src['type'] == 'camera':
            dev = src['dev']
            if dev is None:                      # 自动扫描（在工作线程内，不卡 UI）
                self.logMsg.emit(tr('log_search_cam'))
                devs = find_cameras()
                if not devs:
                    self.logMsg.emit(tr('log_cam_none'))
                    self.camFailed.emit()
                    return
                dev = devs[0]
            self._play_camera(dev)
        else:
            self._play_video(src['path'])

    # ---- 视频文件模式 ----
    def _play_video(self, path):
        path = resolve_video(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.logMsg.emit(tr('log_cannot_open', path=path))
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = Tracker()
        sess = Session(DATA_DIR, os.path.basename(path))
        frame_holder = {}

        def read_fn(kpt8):
            f = frame_holder['f']
            text, conf = self.lpr.read(f, kpt8)
            m = cv2.getPerspectiveTransform(kpt8.astype(np.float32), DST)
            plate = cv2.warpPerspective(f, m, (94, 24))
            return text, conf, plate

        frame_idx = -1
        next_proc_idx = 0
        ema_proc = None
        n_proc = 0
        t_start = time.perf_counter()
        src_name = os.path.basename(path)
        self.logMsg.emit(tr('log_start_video', name=src_name))
        natural = False
        try:
            while not self._stop:
                if self._restart:          # 新源/新会话：退出归档后由 run() 重启
                    break
                if self.paused:
                    self.msleep(50)
                    continue
                ok, frame = cap.read()
                frame_idx += 1
                if not ok:
                    natural = True
                    break
                if frame_idx < next_proc_idx:
                    continue

                frame_holder['f'] = frame
                t0 = time.perf_counter()
                dets = self.det.detect(frame)
                video_t = frame_idx / fps
                events, closed = tracker.step(frame_idx, video_t, dets, read_fn)
                proc = time.perf_counter() - t0
                ema_proc = proc if ema_proc is None else 0.8 * ema_proc + 0.2 * proc
                n_proc += 1

                # realtime 自适应跳帧（约 14fps 分析）
                step = max(1, int(ema_proc * fps + 0.999)) if ema_proc else 1
                next_proc_idx = frame_idx + step

                for ev in events:
                    ui_ev = dict(ev)
                    # 跨线程槽异步执行，必须给 UI 一份独立副本（原 ev 随即被
                    # Session.add_event pop 掉 best_plate，否则缩略图取到 None）
                    if ev.get('best_plate') is not None:
                        ui_ev['_pix'] = ev['best_plate'].copy()
                    sess.add_event(ev)
                    self.eventConfirmed.emit(ui_ev)
                for ct in closed:
                    if not ct.confirmed and ct.total_reads + ct.low_quality_reads >= 2:
                        sess.add_unresolved(ct)

                snaps = tracker.snapshot()
                wall = time.perf_counter() - t_start
                stats = {
                    'fps': n_proc / wall if wall else 0,
                    'frame': frame_idx, 'total': total, 't': video_t,
                    'confirmed': len(sess.events),
                    'candidates': sum(1 for s in snaps if s['state'] == 'candidate'),
                    'det_ms': ema_proc * 1000 if ema_proc else 0,
                    'src': src_name,
                }
                self.frameReady.emit(bgr_to_qimg(frame), snaps, stats)
        finally:
            cap.release()
            json_path, _ = sess.export()
        self.logMsg.emit(tr('log_session_done', n=len(sess.events), path=json_path))
        if natural:
            self.videoFinished.emit(sess.summary(), json_path)

    # ---- 摄像头实时模式 ----
    def _play_camera(self, dev):
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        opened = cap.open(dev, cv2.CAP_V4L2)
        if not opened:
            self.logMsg.emit(tr('log_cannot_open', path=dev))
            self.camFailed.emit()
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # 低延迟：只留最新帧
        for w, h in ((1280, 720), (960, 540), (640, 480)):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w:
                break
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logMsg.emit(tr('log_cam_open', dev=dev, w=w, h=h))

        tracker = Tracker()
        sess = Session(DATA_DIR, time.strftime('camera_%Y%m%d_%H%M%S'))
        frame_holder = {}

        def read_fn(kpt8):
            f = frame_holder['f']
            text, conf = self.lpr.read(f, kpt8)
            m = cv2.getPerspectiveTransform(kpt8.astype(np.float32), DST)
            plate = cv2.warpPerspective(f, m, (94, 24))
            return text, conf, plate

        n_proc = 0
        fail = 0
        ema_proc = None
        lost = False
        t_start = time.perf_counter()
        self.logMsg.emit(tr('log_start_cam'))
        try:
            while not self._stop:
                if self._restart:          # 新源/新会话：退出归档后由 run() 重启
                    break
                if self.paused:
                    if self.export_requested:
                        self.export_requested = False
                        self._export_now(sess)
                    self.msleep(50)
                    continue
                ok, frame = cap.read()
                if not ok:
                    fail += 1
                    if fail >= 30:
                        lost = True
                        break
                    continue
                fail = 0

                frame_holder['f'] = frame
                t0 = time.perf_counter()
                dets = self.det.detect(frame)
                wall_now = time.perf_counter() - t_start
                events, closed = tracker.step(n_proc, wall_now, dets, read_fn)
                proc = time.perf_counter() - t0
                ema_proc = proc if ema_proc is None else 0.8 * ema_proc + 0.2 * proc
                n_proc += 1

                for ev in events:
                    ui_ev = dict(ev)
                    if ev.get('best_plate') is not None:
                        ui_ev['_pix'] = ev['best_plate'].copy()
                    sess.add_event(ev)
                    self.eventConfirmed.emit(ui_ev)
                for ct in closed:
                    if not ct.confirmed and ct.total_reads + ct.low_quality_reads >= 2:
                        sess.add_unresolved(ct)

                if self.export_requested:        # 摄像头模式：随时导出当前会话
                    self.export_requested = False
                    self._export_now(sess)

                snaps = tracker.snapshot()
                stats = {
                    'fps': n_proc / wall_now if wall_now else 0,
                    'frame': n_proc, 'total': 0, 't': wall_now,
                    'confirmed': len(sess.events),
                    'candidates': sum(1 for s in snaps if s['state'] == 'candidate'),
                    'det_ms': ema_proc * 1000 if ema_proc else 0,
                    'src': os.path.basename(dev) if isinstance(dev, str) else f'camera{dev}',
                }
                self.frameReady.emit(bgr_to_qimg(frame), snaps, stats)
        finally:
            cap.release()
            json_path, _ = sess.export()
        self.logMsg.emit(tr('log_session_done', n=len(sess.events), path=json_path))
        if lost:
            self.logMsg.emit(tr('log_cam_lost'))
            self.videoFinished.emit(sess.summary(), json_path)

    def _export_now(self, sess):
        json_path, _ = sess.export()
        self.logMsg.emit(tr('log_exported', path=json_path))
        self.exportDone.emit(json_path)


class VideoWidget(QWidget):
    """视频显示控件：等比缩放渲染帧 + QPainter 叠加三状态框与中文读数"""

    def __init__(self):
        super().__init__()
        self._img = None
        self._snaps = []
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0a101d;border-radius:12px;")
        # 检测框标签用等宽字体（科技感读出；汉字由 fontconfig 回退 CJK 字体）
        self.font_label = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.font_label.setPointSize(10)
        self.font_label.setBold(True)

    def update_frame(self, qimg, snaps):
        self._img = qimg
        self._snaps = snaps
        self.update()

    @staticmethod
    def _brackets(p, r, col, w=2.6):
        """四角瞄准括号（AI 检测锁定框的标志性画法）"""
        arm = max(8.0, min(20.0, min(r.width(), r.height()) * 0.28))
        pen = QPen(col, w)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for cx, cy, sx, sy in ((r.left(), r.top(), 1, 1),
                               (r.right(), r.top(), -1, 1),
                               (r.left(), r.bottom(), 1, -1),
                               (r.right(), r.bottom(), -1, -1)):
            p.drawLine(QPointF(cx, cy), QPointF(cx + sx * arm, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + sy * arm))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._img is None:
            # 待机：中央占位文案 + 四角暗青括号（等待信号画面）
            p.setPen(QColor(100, 116, 139))
            f = QFont("Noto Sans CJK SC, 15")
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, tr('placeholder'))
            self._brackets(p, QRectF(self.rect()).adjusted(10, 10, -10, -10),
                           QColor(56, 130, 210, 110), w=2)
            return

        pw, ph = self._img.width(), self._img.height()
        scale = min(self.width() / pw, self.height() / ph)
        dw, dh = pw * scale, ph * scale
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        tgt = QRectF(ox, oy, dw, dh)
        p.drawImage(tgt, self._img)

        # 画面发光边框：三层青色描边由外向内渐亮（监控大屏氛围光）
        p.setBrush(Qt.NoBrush)
        for wdt, adj, alpha in ((13, 6.5, 22), (7, 3.5, 45), (3, 1.5, 90)):
            pen = QPen(QColor(34, 211, 238, alpha))
            pen.setWidth(wdt)
            p.setPen(pen)
            p.drawRect(tgt.adjusted(-adj, -adj, adj, adj))

        for s in self._snaps:
            x1, y1, x2, y2 = s['box']
            r = QRectF(ox + x1 * scale, oy + y1 * scale,
                       (x2 - x1) * scale, (y2 - y1) * scale)
            if s['state'] == 'detected':
                # 检测中：雷达式虚线框
                col, tag = COL_DETECTED, tr('state_detected')
                pen = QPen(QColor(col.red(), col.green(), col.blue(), 170), 1.6)
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawRect(r)
            elif s['state'] == 'candidate':
                # 候选中：琥珀实线框 + 角括号
                col = COL_CANDIDATE
                tag = tr('state_candidate', text=s['text'],
                         votes=s['votes'], need=s['need'])
                p.setPen(QPen(col, 1.8, Qt.SolidLine))
                p.drawRect(r)
                self._brackets(p, r, col, w=2.4)
            else:
                # 已确认：外发光 + 实心角括号（锁定完成）
                col = COL_CONFIRMED
                tag = s['text']
                glow = QColor(col)
                glow.setAlpha(80)
                p.setPen(QPen(glow, 6))
                p.drawRect(r)
                self._brackets(p, r, col, w=2.8)
            # 标签底色块 + 文本
            p.setFont(self.font_label)
            tw = p.fontMetrics().horizontalAdvance(tag) + 12
            th = p.fontMetrics().height() + 4
            ty = max(oy, r.top() - th - 2)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(r.left(), ty, tw, th))
            p.setPen(COL_TAGTEXT)
            p.drawText(QRectF(r.left() + 6, ty, tw - 12, th),
                       Qt.AlignVCenter | Qt.AlignLeft, tag)


class SpotCard(QFrame):
    """上客区单个车位卡：车位号 + 状态文字 + 车牌（大字）"""

    STATE_COL = {'free': '#64748b', 'assigned': '#fbbf24',
                 'serving': '#34d399', 'timeout': '#f87171'}

    def __init__(self, name):
        super().__init__()
        self.setObjectName('spotcard')
        self.name = name
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.lab_name = QLabel(name)
        self.lab_name.setObjectName('spotname')
        self.lab_state = QLabel()
        self.lab_plate = QLabel('—')
        self.lab_plate.setObjectName('spotplate')
        self.lab_plate.setMinimumHeight(22)
        lay.addWidget(self.lab_name)
        lay.addWidget(self.lab_state)
        lay.addWidget(self.lab_plate)
        self.show_state('free', None)

    def show_state(self, state, plate, timed_out=False):
        key = 'timeout' if (timed_out and state == 'assigned') else state
        self.lab_state.setText(tr('ost_timeout') if key == 'timeout'
                               else tr('spot_' + state))
        self.lab_state.setStyleSheet(
            f"color:{self.STATE_COL[key]};font-size:13px;font-weight:700;")
        self.lab_plate.setText(plate or '—')


class QueueCard(QFrame):
    """候车队列单行：车牌 + 状态 + 车位 + 操作按钮（到位/完成/取消/补登记）"""

    def __init__(self, kind, plate, oid, state, spot, timed_out,
                 on_arrive, on_finish, on_cancel, on_register):
        super().__init__()
        self.setObjectName('queuecard')
        st = kind if kind == 'unreg' else (
            'timeout' if (timed_out and state == ORDER_ASSIGNED) else state)
        self.setProperty('st', st)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)
        self.lab_plate = QLabel(plate)
        self.lab_plate.setObjectName('qplate')
        lay.addWidget(self.lab_plate)
        lay.addSpacing(6)
        self.lab_state = QLabel()
        self.lab_state.setObjectName('qstate')
        lay.addWidget(self.lab_state)
        lay.addStretch()
        self.btn_a = QPushButton()
        self.btn_a.setObjectName('qbtn')
        self.btn_a.setCursor(Qt.PointingHandCursor)
        self.btn_b = QPushButton()
        self.btn_b.setObjectName('qbtn')
        self.btn_b.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_a)
        lay.addWidget(self.btn_b)
        if kind == 'unreg':
            self.lab_state.setText(tr('unreg_row'))
            self.btn_a.hide()
            self.btn_b.setText(tr('btn_regen'))
            self.btn_b.clicked.connect(lambda: on_register(plate))
        else:
            txt = tr('ost_wait') if state == ORDER_WAITING else \
                tr('spot_assigned' if state == ORDER_ASSIGNED else 'spot_serving')
            self.lab_state.setText(txt + (f' · {spot}' if spot else ''))
            if state == ORDER_ASSIGNED:
                self.btn_a.setText(tr('btn_arrive'))
                self.btn_a.clicked.connect(lambda: on_arrive(oid))
                self.btn_b.setText(tr('btn_cancel'))
                self.btn_b.clicked.connect(lambda: on_cancel(oid))
            elif state == ORDER_SERVING:
                self.btn_a.hide()
                self.btn_b.setText(tr('btn_finish'))
                self.btn_b.clicked.connect(lambda: on_finish(oid))
            else:   # waiting
                self.btn_a.hide()
                self.btn_b.setText(tr('btn_cancel'))
                self.btn_b.clicked.connect(lambda: on_cancel(oid))


class MainWindow(QMainWindow):
    # 语音开播 → 切横幅（携带该条自己的文案，积压也不会错位；voice 线程安全发射）
    bannerAdvance = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr('app_title'))
        self.resize(1280, 820)
        self.worker = Worker()
        self.worker.frameReady.connect(self.on_frame)
        self.worker.eventConfirmed.connect(self.on_event)
        self.worker.logMsg.connect(lambda m: self.stat.setText(m))
        self.worker.videoFinished.connect(self.on_finished)
        self.worker.exportDone.connect(self.on_export_done)
        self.worker.camFailed.connect(self.on_cam_failed)
        # 调度核心：识别确认事件 → on_plate_seen；现场按钮 → arrive/complete/cancel
        self.dispatch = DispatchCore()
        self.unreg_plates = {}     # plate -> 最近识别时间（未登记车辆行）
        # 横幅播报机制：语音开着时每条播报开播回调直接切横幅（完全同步）；
        # 语音关闭时文案入此队，由 _tick 定时推进
        self.banner_pending = deque()    # 仅 PLATE_VOICE=0 时使用 [(text, urgent)]
        self.hist = deque()             # 历史条最近 3 条
        self._banner_t = 0.0
        os.makedirs(DATA_DIR, exist_ok=True)
        self.journal_path = os.path.join(DATA_DIR, 'dispatch_log.jsonl')
        self.voice = VoicePlayer(DATA_DIR)
        self.last_json = None
        self._last_stats = None
        self._last_snaps = []
        self._banner_touched = False
        self._build_ui()
        self.retranslate()
        # 语音开播回调在 voice 线程触发 → 类信号自动队列到 UI 线程切横幅
        self.bannerAdvance.connect(self._on_banner_advance)
        # 超时巡检：每秒 tick 一次（真实时钟）
        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self._tick)
        self.tick_timer.start(1000)
        self.voice.say('welcome')

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(14, 12, 14, 12)
        root_lay.setSpacing(12)

        # ---- 顶栏：品牌点 + 标题 + 源信息 + 调度统计 + 语言切换 ----
        top = QFrame()
        top.setObjectName('panel')
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 10, 16, 10)
        tl.setSpacing(12)
        dot = QLabel()
        dot.setObjectName('dot')
        dot.setFixedSize(10, 10)
        self.brand = QLabel()
        self.brand.setObjectName('brand')
        self.meta = QLabel()
        self.meta.setObjectName('meta')
        self.count_lab = QLabel()
        self.count_lab.setObjectName('count')
        self.stat = QLabel()
        self.stat.setObjectName('stat')
        self.btn_lang = QPushButton()
        self.btn_lang.setObjectName('lang')
        self.btn_lang.setCursor(Qt.PointingHandCursor)
        self.btn_lang.clicked.connect(self.toggle_lang)
        tl.addWidget(dot)
        tl.addWidget(self.brand)
        tl.addSpacing(16)
        tl.addWidget(self.meta, 1)
        tl.addWidget(self.count_lab)
        tl.addSpacing(12)
        tl.addWidget(self.stat)
        tl.addWidget(self.btn_lang)
        root_lay.addWidget(top)

        # ---- 主体：左侧识别画面 + 右侧上客区 ----
        body = QHBoxLayout()
        body.setSpacing(12)
        self.video = VideoWidget()
        body.addWidget(self.video, 3)

        right_widget = QWidget()
        right_widget.setFixedWidth(430)
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)

        # 大字播报横幅
        self.banner = QLabel()
        self.banner.setObjectName('banner')
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setMinimumHeight(64)
        self.banner.setWordWrap(True)
        right.addWidget(self.banner)

        # 最近事件历史条（小字）：横幅一次只能显示一条语音事件，
        # 同时发生的事件（完成+补位/批量分配）在这里全部可见
        self.hist_lab = QLabel()
        self.hist_lab.setObjectName('hist')
        right.addWidget(self.hist_lab)

        # 上客区面板：2x2 车位卡
        self.pickup_box = QFrame()
        self.pickup_box.setObjectName('panel')
        pl = QVBoxLayout(self.pickup_box)
        pl.setContentsMargins(12, 10, 12, 12)
        self.pickup_cap = QLabel()
        self.pickup_cap.setObjectName('panelcap')
        pl.addWidget(self.pickup_cap)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.spot_cards = {}
        for i, name in enumerate(self.dispatch.spot_names):
            card = SpotCard(name)
            self.spot_cards[name] = card
            grid.addWidget(card, i // 2, i % 2)
        pl.addLayout(grid)
        right.addWidget(self.pickup_box)
        right.addStretch()
        body.addWidget(right_widget, 0)
        root_lay.addLayout(body, 1)

        # ---- 下方：候车队列 ----
        self.queue_box = QFrame()
        self.queue_box.setObjectName('panel')
        ql = QVBoxLayout(self.queue_box)
        ql.setContentsMargins(12, 10, 12, 12)
        ql.setSpacing(8)
        qhead = QHBoxLayout()
        self.queue_cap = QLabel()
        self.queue_cap.setObjectName('panelcap')
        qhead.addWidget(self.queue_cap)
        qhead.addStretch()
        self.btn_register = QPushButton()
        self.btn_register.setObjectName('primary')
        self.btn_register.setMinimumHeight(34)
        self.btn_preset = QPushButton()
        self.btn_preset.setMinimumHeight(34)
        self.btn_demo = QPushButton()
        self.btn_demo.setMinimumHeight(34)
        qhead.addWidget(self.btn_demo)
        qhead.addWidget(self.btn_preset)
        qhead.addWidget(self.btn_register)
        ql.addLayout(qhead)

        self.queue_host = QWidget()
        self.queue_lay = QVBoxLayout(self.queue_host)
        self.queue_lay.setContentsMargins(0, 0, 0, 0)
        self.queue_lay.setSpacing(6)
        self.queue_lay.addStretch()
        qscroll = QScrollArea()
        qscroll.setWidgetResizable(True)
        qscroll.setWidget(self.queue_host)
        qscroll.setMaximumHeight(190)
        qscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ql.addWidget(qscroll, 1)
        root_lay.addWidget(self.queue_box)

        # ---- 底栏：识别源操作 ----
        bot = QHBoxLayout()
        bot.setSpacing(10)
        self.btn_open = QPushButton()
        self.btn_open.setObjectName('primary')
        self.btn_camera = QPushButton()
        self.btn_camera.setObjectName('primary')
        self.btn_pause = QPushButton()
        self.btn_restart = QPushButton()
        self.btn_export = QPushButton()
        for b in (self.btn_open, self.btn_camera, self.btn_pause,
                  self.btn_restart, self.btn_export):
            b.setMinimumHeight(40)
        self.btn_open.setMinimumWidth(116)
        self.btn_camera.setMinimumWidth(132)
        bot.addWidget(self.btn_open)
        bot.addWidget(self.btn_camera)
        bot.addWidget(self.btn_pause)
        bot.addWidget(self.btn_restart)
        bot.addWidget(self.btn_export)
        bot.addStretch()
        root_lay.addLayout(bot)

        self.btn_open.clicked.connect(self.pick_video)
        self.btn_camera.clicked.connect(self.do_camera)
        self.btn_pause.clicked.connect(self.do_pause)
        self.btn_restart.clicked.connect(self.do_restart)
        self.btn_export.clicked.connect(self.do_export)
        # clicked 自带 checked=False 实参，直连会挤掉 plate=None 默认值导致不弹输入框
        self.btn_register.clicked.connect(lambda: self.do_register())
        self.btn_preset.clicked.connect(self.do_preset)
        self.btn_demo.clicked.connect(self.start_demo)   # 按钮进预演（start_demo 自带防重入）
        self.btn_pause.setEnabled(False)
        self.btn_restart.setEnabled(False)
        self.btn_export.setEnabled(False)

    # ---- 语言切换 ----
    def toggle_lang(self):
        set_lang('en' if LANG == 'zh' else 'zh')
        self.retranslate()

    def retranslate(self):
        """切换语言后刷新所有静态文案（动态文案随下一帧/下一条日志自动更新）"""
        self.setWindowTitle(tr('app_title'))
        self.brand.setText(tr('app_title'))
        self.btn_lang.setText('中文' if LANG == 'en' else 'EN')
        self.btn_open.setText(tr('btn_open'))
        self.btn_camera.setText(tr('btn_camera'))
        self.btn_pause.setText(
            tr('btn_resume') if self.worker.paused else tr('btn_pause'))
        self.btn_restart.setText(tr('btn_restart'))
        self.btn_export.setText(tr('btn_export'))
        self.btn_register.setText(tr('btn_register'))
        self.btn_preset.setText(tr('btn_preset'))
        self.btn_demo.setText(tr('btn_demo'))
        self.pickup_cap.setText(tr('zone_pickup'))
        self.queue_cap.setText(tr('zone_queue'))
        if not self._banner_touched:
            self.banner.setText(tr('banner_idle'))
        if self._last_stats is None:
            self.meta.setText(tr('meta_idle'))
            self.stat.setText(tr('stat_ready'))
        else:
            self._update_meta(self._last_stats)
        self._refresh_dispatch()
        self.video.update()

    # ---- 识别源 ----
    def pick_video(self):
        dlg = QFileDialog(self, tr('dlg_pick_video'), VIDEO_DIR,
                          "Video (*.mp4 *.mov *.avi *.mkv)")
        # 板上原生 GTK 对话框不响应 QSS（深底黑字看不清），强制 Qt 渲染
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if not dlg.exec() or not dlg.selectedFiles():
            return
        path = dlg.selectedFiles()[0]
        self._reset_recognition()
        self.worker.open_video(path)
        self._enable_run_buttons()
        self.btn_pause.setText(tr('btn_pause'))

    def do_camera(self):
        self._reset_recognition()
        self.worker.open_camera(None)
        self._enable_run_buttons()
        self.btn_pause.setText(tr('btn_pause'))

    def _reset_recognition(self):
        """切换识别源只重置识别统计；调度订单跨源保留（视频↔摄像头同一套队列）"""
        self._last_stats = None
        self._last_snaps = []
        self.meta.setText(tr('meta_idle'))
        self.last_json = None

    def _enable_run_buttons(self):
        self.btn_pause.setEnabled(True)
        self.btn_restart.setEnabled(True)
        self.btn_export.setEnabled(True)

    def do_pause(self):
        self.worker.toggle_pause()
        self.btn_pause.setText(
            tr('btn_resume') if self.worker.paused else tr('btn_pause'))

    def do_restart(self):
        self._reset_recognition()
        self.worker.restart_session()
        self.btn_pause.setText(tr('btn_pause'))

    def _info_box(self, title, text, warning=False):
        """深色主题消息框：parent 主窗居中、按钮本地化（QSS 统一样式）"""
        box = QMessageBox(QMessageBox.Warning if warning else QMessageBox.Information,
                          title, text, parent=self)
        box.addButton(tr('btn_ok'), QMessageBox.AcceptRole)
        box.exec()

    def do_export(self):
        src = self.worker.source
        if src and src['type'] == 'camera' and self.worker.isRunning():
            self.worker.export_requested = True    # 摄像头：线程内即时导出
        elif self.last_json:
            self._info_box(tr('dlg_export'),
                           tr('dlg_exported', path=self.last_json))
        else:
            self.stat.setText(tr('stat_wait_end'))

    def export_dispatch_snapshot(self):
        """调度快照：全部订单 + 统计导出为 JSON（识别会话导出后自动附带调用）"""
        d = self.dispatch
        snap = {'wall': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': d.stats(),
                'journal': self.journal_path,
                'orders': [{'oid': o.oid, 'plate': o.plate, 'state': o.state,
                            'spot': o.spot, 'timed_out': o.timed_out,
                            'registered_t': o.registered_t,
                            'assigned_t': o.assigned_t, 'arrived_t': o.arrived_t,
                            'done_t': o.done_t}
                           for o in sorted(d.orders.values(), key=lambda x: x.oid)]}
        path = os.path.join(DATA_DIR,
                            f'dispatch_snapshot_{time.strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(snap, f, ensure_ascii=False, indent=1)
            print(f'[dispatch] snapshot saved: {path}', flush=True)
            return path
        except OSError as ex:
            print(f'[dispatch] snapshot FAILED: {ex}', flush=True)
            return None

    # ---- 调度：登记 ----
    def do_register(self, plate=None):
        if plate is None:
            plate, ok = QInputDialog.getText(self, tr('dlg_reg_title'),
                                              tr('dlg_reg_hint'))
            if not ok:
                return
        plate = plate.strip().upper()
        if not (7 <= len(plate) <= 8):
            self._info_box(tr('dlg_reg_title'), tr('dlg_reg_bad'), warning=True)
            return
        evs = self.dispatch.register(plate, time.time())
        self.unreg_plates.pop(plate, None)
        for e in evs:
            if e['type'] == 'register_rejected':
                self._info_box(tr('dlg_reg_title'),
                               tr('dlg_reg_dup', plate=plate), warning=True)
                return
        self._apply_events(evs)

    def do_preset(self):
        """预置演示车辆：登记测试视频里的真实车牌（播放即真实触发）"""
        evs = []
        for p in PRESET_VEHICLES:
            evs += self.dispatch.register(p, time.time())
            self.unreg_plates.pop(p, None)
        self._apply_events(evs)

    # ---- 调度：现场按钮 ----
    def _do_arrive(self, oid):
        self._apply_events(self.dispatch.arrive(oid, time.time()))

    def _do_finish(self, oid):
        self._apply_events(self.dispatch.complete(oid, time.time()))

    def _do_cancel(self, oid):
        self._apply_events(self.dispatch.cancel(oid, time.time()))

    def _tick(self):
        """每秒超时巡检（真实时钟，与识别源无关）"""
        self._apply_events(self.dispatch.tick(time.time()))
        # 语音关闭时（PLATE_VOICE=0）没有开播回调，横幅队列用定时节奏推进
        if (not self.voice.enabled and self.banner_pending
                and time.time() - self._banner_t >= 2.8):
            text, urgent = self.banner_pending.popleft()
            self._set_banner(text, urgent=urgent)

    # ---- 预演模式：全自动跑通调度全流程 ----
    # 覆盖全部情况：登记分配/车位满排队/FIFO补位/重复登记拒绝/到位/完成/
    # 超时警示/取消释放/二次接客；登记按 DEMO_SCHEDULE 错峰触发（语音不积压），
    # 未登记提示与 reannounce·等待节流·serving 静默由视频识别自然触发。
    DEMO_ARRIVE_S = 8.0     # 已分配 N 秒后模拟车辆开进车位
    DEMO_SERVE_S = 10.0     # 到位 N 秒后模拟接客完成、释放车位
    DEMO_REG_S = 3.0        # 发现未登记 N 秒后自动补登记
    DEMO_TIMEOUT_S = 30.0   # 演示用超时窗口（真实默认 120s 等太久）
    DEMO_CANCEL_S = 5.0     # 超时红显 N 秒后自动取消，演示"取消释放+补位"
    DEMO_TIMEOUT_PLATE = '苏A0AR19'   # 这辆故意不到位，专门触发超时分支

    # ---- 演示视频录制（--record：帧序列 + meta.json 事件时间线）----
    def start_record(self, rdir):
        """10fps 定时 grab() 存帧；语音/调度事件按墙上时钟记入 meta.json。
        语音拼接 wav 由 voice.keep 保留在 data/_voice/，合成时按 meta 混音。"""
        os.makedirs(rdir, exist_ok=True)
        self._rec_dir = rdir
        self._rec_t0 = time.time()
        self._rec_n = 0
        self._rec_audio = []
        self._rec_events = []
        self.voice.keep = True
        self.voice.on_play = self._on_voice_played
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._rec_frame)
        self._rec_timer.start(100)                  # RECORD_FPS = 10
        print(f"[record] -> {rdir} @ 10fps", flush=True)

    def _rec_frame(self):
        self._rec_n += 1
        # 弹层/对话框均为窗口子控件，self.grab() 直接包含
        self.grab().save(os.path.join(self._rec_dir, 'frame_%05d.jpg' % self._rec_n))

    def _on_voice_played(self, t, tokens, wav):
        # voice 线程触发；append 原子性足够，meta 落盘在退出时
        self._rec_audio.append(
            {'t': round(t - self._rec_t0, 3), 'tokens': tokens, 'wav': wav})

    def write_rec_meta(self):
        if not getattr(self, '_rec_dir', None):
            return
        meta = {'record_fps': 10, 'frames': self._rec_n,
                'audio_events': self._rec_audio,
                'dispatch_events': self._rec_events}
        with open(os.path.join(self._rec_dir, 'meta.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"[record] done: {self._rec_n} frames, "
              f"{len(self._rec_audio)} voice events, "
              f"{len(self._rec_events)} dispatch events -> meta.json", flush=True)

    def start_demo(self):
        # 防重入：预演已在进行中再点按钮直接忽略（--demo 启动与按钮共用此入口）
        if getattr(self, 'demo_timer', None) is not None and \
                self.demo_timer.isActive():
            return
        # 换短超时窗口的调度核心（预置前替换，登记/分配全走新核心）
        # 时序可用环境变量压缩（演示视频录制用，默认保持产品节奏不变）
        _env = lambda k, d: float(os.environ.get(k, d))
        self.DEMO_ARRIVE_S = _env('PLATE_DEMO_ARRIVE_S', self.DEMO_ARRIVE_S)
        self.DEMO_SERVE_S = _env('PLATE_DEMO_SERVE_S', self.DEMO_SERVE_S)
        self.DEMO_REG_S = _env('PLATE_DEMO_REG_S', self.DEMO_REG_S)
        self.DEMO_TIMEOUT_S = _env('PLATE_DEMO_TIMEOUT_S', self.DEMO_TIMEOUT_S)
        self.DEMO_CANCEL_S = _env('PLATE_DEMO_CANCEL_S', self.DEMO_CANCEL_S)
        self.dispatch = DispatchCore(timeout_s=self.DEMO_TIMEOUT_S)
        self.demo_second_done = False
        # 错峰登记：按 DEMO_SCHEDULE 计划表在 _demo_tick 里逐辆登记，
        # 避免一次性登记造成语音播报积压（演示视频音画同步）
        self.demo_t0 = time.time()
        self.demo_sched = list(DEMO_SCHEDULE)
        self._set_banner(tr('banner_demo'))
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self._demo_tick)
        self.demo_timer.start(1000)

    def _demo_tick(self):
        now = time.time()
        # 0) 登记计划表：错峰触发登记（含候队/重复拒绝演示条目）。
        #    直接走 dispatch.register + _apply_events：重复登记只上横幅，
        #    不弹 do_register 的模态框，预演流程不被打断
        while self.demo_sched and now - self.demo_t0 >= self.demo_sched[0][1]:
            plate, _ = self.demo_sched.pop(0)
            evs = self.dispatch.register(plate, now)
            self.unreg_plates.pop(plate, None)
            self._apply_events(evs)
        # 1) 未登记车辆自动补登记（演示「补登记进队列」分支）
        for plate in list(self.unreg_plates):
            if now - self.unreg_plates[plate] >= self.DEMO_REG_S:
                self.do_register(plate)
        # 2) 超时车辆红显 N 秒后自动取消（车位释放 → FIFO 补位）
        o = self.dispatch.active_by_plate.get(self.DEMO_TIMEOUT_PLATE)
        if o is not None and o.timed_out and now - o.assigned_t >= \
                self.DEMO_TIMEOUT_S + self.DEMO_CANCEL_S:
            self._apply_events(self.dispatch.cancel(o.oid, now))
        # 3) 模拟物理动作：到位（开进车位）→ 完成（乘客上车驶离）；
        #    超时演示车故意不到，其余 assigned 满 8s 到位、serving 满 10s 完成
        for o in list(self.dispatch.orders.values()):
            if o.state == ORDER_ASSIGNED:
                if o.plate == self.DEMO_TIMEOUT_PLATE:
                    continue                  # 留给 tick 超时红显
                if now - o.assigned_t >= self.DEMO_ARRIVE_S:
                    self._apply_events(self.dispatch.arrive(o.oid, now))
            elif o.state == ORDER_SERVING and now - o.arrived_t >= self.DEMO_SERVE_S:
                self._apply_events(self.dispatch.complete(o.oid, now))
        # 4) 全部流转完（有完成记录且无活跃订单/占用）→ 演示「二次接客」一轮：
        #    第一辆完成的车重新登记，再走完整分配→到位→完成流程
        st = self.dispatch.stats()
        if (not self.demo_second_done and st['done'] > 0
                and st['waiting'] == 0 and st['serving'] == 0
                and st['spots_used'] == 0):
            self.demo_second_done = True
            self._apply_events(self.dispatch.register('苏A53D9P', now))

    # ---- 调度：事件应用与渲染 ----
    BANNER_BY_TYPE = {'assigned': 'banner_assign', 'arrived': 'banner_arrived',
                      'completed': 'banner_done', 'cancelled': 'banner_cancel',
                      'unregistered': 'banner_unreg', 'reannounce': 'banner_assign',
                      'still_waiting': 'banner_wait', 'timeout': 'banner_timeout',
                      'register_rejected': 'banner_dup'}

    # 调度事件 → 语音词元（只播关键调度动作；未登记/等待中仅视觉提示，不抢广播）
    # with_plate: True=车牌+短语+车位；'arrived'=到位专用语序（车位嵌句中）；False=只播短语
    VOICE_BY_TYPE = {'assigned': ('please_go', True), 'reannounce': ('please_go', True),
                     'arrived': (None, 'arrived'),
                     'completed': ('finished', False), 'cancelled': ('cancelled', False),
                     'timeout': ('timeout', True)}

    # 历史条短语 key（最近 3 条小字，全事件可见——补横幅一次只能一条的不足）
    HIST_BY_TYPE = {'registered': 'hist_registered', 'assigned': 'hist_assigned',
                    'reannounce': 'hist_reannounce', 'arrived': 'hist_arrived',
                    'completed': 'hist_completed', 'cancelled': 'hist_cancelled',
                    'timeout': 'hist_timeout', 'unregistered': 'hist_unregistered',
                    'still_waiting': 'hist_still_waiting',
                    'register_rejected': 'hist_register_rejected'}

    def _apply_events(self, evs):
        if not evs:
            return
        if getattr(self, '_rec_dir', None):
            # 录制模式：调度事件时间线（type/plate/spot）供演示视频对时
            self._rec_events.append({
                't': round(time.time() - self._rec_t0, 3),
                'evs': [{'type': e['type'], 'plate': e.get('plate'),
                         'spot': e.get('spot')} for e in evs]})
        self._journal(evs)
        for e in evs:
            self._add_hist(e)
            key = self.BANNER_BY_TYPE.get(e['type'])
            text = tr(key, **e) if key else ''
            urgent = e['type'] in ('timeout', 'unregistered')
            v = self.VOICE_BY_TYPE.get(e['type'])
            if v:
                phrase, with_plate = v
                if text:
                    if self.voice.enabled:
                        # 每条播报闭包携带自己的文案：开播即切，积压不错位
                        cb = (lambda t, u: lambda: self.bannerAdvance.emit(t, u))(
                            text, urgent)
                    else:
                        # 语音关闭：无回调，横幅走定时节奏（_tick 推进）
                        self.banner_pending.append((text, urgent))
                        cb = None
                else:
                    cb = None
                if with_plate == 'arrived':
                    # 到位语序：车牌+已到位+A02+开始接客（车位嵌句中，不能落句尾）
                    self.voice.announce_arrived(e['plate'], e.get('spot'),
                                                on_start=cb)
                elif with_plate:
                    # 先车牌后指令（真实调度广播惯例）: 「苏A53D9P，请前往 A01」
                    self.voice.announce_plate(e['plate'], phrase, e.get('spot'),
                                              on_start=cb)
                else:
                    self.voice.say(phrase, on_start=cb)
            elif text and not self.banner_pending and self.voice.q.empty():
                # 无语音事件（未登记/等待/重复拒绝）：横幅空闲才直接亮，
                # 有播报排队时不抢屏——反正历史条已记录
                self._set_banner(text, urgent=urgent)
            if e['type'] == 'unregistered':
                self.unreg_plates[e['plate']] = time.time()
        self._refresh_dispatch()

    def _on_banner_advance(self, text, urgent):
        """语音开播信号（UI 线程）：横幅切到正在播的这条，完全同步"""
        self._set_banner(text, urgent=urgent)

    def _add_hist(self, e):
        key = self.HIST_BY_TYPE.get(e['type'])
        if not key:
            return
        try:
            txt = tr(key, **e)
        except Exception:
            txt = e['plate']
        self.hist.appendleft(f"{e.get('t', '')} {txt}")
        while len(self.hist) > 3:                # 只留最近 3 条
            self.hist.pop()
        self.hist_lab.setText('  ·  '.join(self.hist))

    def _journal(self, evs):
        """调度日志：逐事件追加 jsonl（登记/分配/到位/完成/取消/超时/未登记）"""
        try:
            with open(self.journal_path, 'a', encoding='utf-8') as f:
                for e in evs:
                    f.write(json.dumps(
                        {'wall': time.strftime('%Y-%m-%d %H:%M:%S'), **e},
                        ensure_ascii=False) + '\n')
        except OSError as ex:
            print(f'[journal] WARN: {ex}', flush=True)

    def _set_banner(self, text, urgent=False):
        self._banner_touched = True
        self._banner_t = time.time()
        self.banner.setText(text)
        self.banner.setProperty('urgent', urgent)
        # 属性选择器需要重新 polish 才生效
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)

    def _refresh_dispatch(self):
        """全量刷新：车位卡 + 队列行 + 顶栏统计（数据量小，重建即可）"""
        d = self.dispatch
        # 车位卡
        order_by_spot = {}
        for o in d.orders.values():
            if o.spot and o.state in (ORDER_ASSIGNED, ORDER_SERVING):
                order_by_spot[o.spot] = o
        for name, card in self.spot_cards.items():
            o = order_by_spot.get(name)
            if o is None:
                card.show_state('free', None)
            else:
                state = 'assigned' if o.state == ORDER_ASSIGNED else 'serving'
                card.show_state(state, o.plate, timed_out=o.timed_out)
        # 队列行（活跃订单在前，未登记在后）
        while self.queue_lay.count():
            it = self.queue_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        rows = [(o.oid, o.plate, o.state, o.spot, o.timed_out)
                for o in d.active_orders()]
        for plate in self.unreg_plates:
            if plate not in d.active_by_plate:
                rows.append((None, plate, 'unreg', None, False))
        if not rows:
            empty = QLabel(tr('queue_empty'))
            empty.setObjectName('qempty')
            self.queue_lay.addWidget(empty)
        else:
            for oid, plate, state, spot, t_out in rows:
                if state == 'unreg':
                    card = QueueCard('unreg', plate, None, None, None, False,
                                     None, None, None, self.do_register)
                else:
                    card = QueueCard('order', plate, oid, state, spot, t_out,
                                     self._do_arrive, self._do_finish,
                                     self._do_cancel, self.do_register)
                self.queue_lay.addWidget(card)
        self.queue_lay.addStretch()
        # 顶栏统计
        s = d.stats()
        self.count_lab.setText(tr('stat_dispatch', done=s['done'],
                                  waiting=s['waiting'], used=s['spots_used'],
                                  total=s['spots_total']))

    # ---- Worker 槽 ----
    def on_frame(self, qimg, snaps, stats):
        self._last_stats = stats
        self._last_snaps = snaps
        self.video.update_frame(qimg, snaps)
        self._update_meta(stats)

    def _update_meta(self, stats):
        name = stats.get('src') or ''
        if stats.get('total'):
            self.meta.setText(tr('meta_video', name=name, fps=stats['fps'],
                                 ms=stats['det_ms'], frame=stats['frame'],
                                 total=stats['total'], t=stats['t']))
        else:
            self.meta.setText(tr('meta_camera', name=name, fps=stats['fps'],
                                  ms=stats['det_ms'], t=stats['t']))

    def on_event(self, ev):
        """识别确认 → 进站信号，驱动调度（视频模拟与摄像头实时共用）"""
        self._apply_events(self.dispatch.on_plate_seen(ev['plate'], time.time()))

    def on_finished(self, summary, json_path):
        self.last_json = json_path
        self.stat.setText(tr('stat_finished'))
        self.btn_pause.setEnabled(False)
        self.export_dispatch_snapshot()

    def on_export_done(self, json_path):
        self.last_json = json_path
        snap = self.export_dispatch_snapshot()
        msg = tr('dlg_exported', path=json_path)
        if snap:
            msg += '\n' + tr('dlg_exported', path=snap)
        self._info_box(tr('dlg_export'), msg)

    def on_cam_failed(self):
        self.btn_pause.setEnabled(False)
        self.btn_restart.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._info_box(tr('dlg_cam_title'), tr('dlg_no_cam'), warning=True)

    def closeEvent(self, e):
        self.write_rec_meta()
        self.voice.shutdown()
        self.worker.stop()
        self.worker.wait(2000)
        e.accept()


def main():
    args = list(sys.argv[1:])
    shot_path = None
    shot_ms = 4000
    positional = []
    want_cam = False
    want_preset = False
    want_demo = False
    reg_plates = []
    cam_dev = None
    record_dir = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--record' and i + 1 < len(args) \
                and not args[i + 1].startswith('--'):  # 演示视频录制：帧序列+meta
            record_dir = args[i + 1]
            i += 2
        elif a == '--record':
            record_dir = os.path.join(APP_ROOT, 'demo_out')
            i += 1
        elif a == '--shot' and i + 1 < len(args):
            shot_path = args[i + 1]
            i += 2
        elif a == '--shot-ms' and i + 1 < len(args):
            shot_ms = int(args[i + 1])
            i += 2
        elif a.startswith('--shot-ms='):
            shot_ms = int(a.split('=', 1)[1])
            i += 1
        elif a == '--camera':
            want_cam = True
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                v = args[i + 1]
                cam_dev = int(v) if v.isdigit() else v
                i += 2
            else:
                i += 1
        elif a.startswith('--camera='):
            want_cam = True
            v = a.split('=', 1)[1]
            cam_dev = int(v) if v.isdigit() else v
            i += 1
        elif a == '--lang' and i + 1 < len(args):
            set_lang(args[i + 1])
            i += 2
        elif a.startswith('--lang='):
            set_lang(a.split('=', 1)[1])
            i += 1
        elif a == '--preset':                     # 启动即预置演示车辆
            want_preset = True
            i += 1
        elif a == '--demo':                       # 预演模式：全自动跑通调度全流程
            want_demo = True
            i += 1
        elif a == '--register' and i + 1 < len(args):   # 启动即登记车牌
            reg_plates.append(args[i + 1])
            i += 2
        else:
            positional.append(a)
            i += 1

    app = QApplication(sys.argv)
    # 深色科技风：Fusion 风格保证跨平台控件行为一致，调色板兜底原生控件
    app.setStyle('Fusion')
    app.setPalette(build_dark_palette())
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    if want_preset and not want_demo:              # --demo 内部含预置，勿重复
        win.do_preset()
    if record_dir:
        win.start_record(record_dir)              # 先于 start_demo，录到启动登记事件
    if want_demo:                                 # 预演：换短超时核心+预置+全自动驱动
        win.start_demo()
    if record_dir and want_demo:
        # 预演录制：二次接客完成（全清零）后定格 3.5s 自动收尾退出
        def _demo_watch():
            if not getattr(win, 'demo_second_done', False):
                return
            st = win.dispatch.stats()
            if st['done'] > 0 and st['waiting'] == 0 and \
                    st['serving'] == 0 and st['spots_used'] == 0:
                watch.stop()
                def _finish():
                    win.write_rec_meta()
                    win.worker.stop()
                    win.worker.wait(1500)
                    win.voice.shutdown()
                    app.quit()
                QTimer.singleShot(3500, _finish)
        watch = QTimer()
        watch.timeout.connect(_demo_watch)
        watch.start(1000)
    for p in reg_plates:
        win.do_register(p)
    if positional:                               # 启动即加载视频
        vpath = resolve_video(positional[0])
        if os.path.isfile(vpath):
            win._reset_recognition()
            win.worker.open_video(vpath)
            win._enable_run_buttons()
    if want_cam:                                 # 启动即进入摄像头模式
        win._reset_recognition()
        win.worker.open_camera(cam_dev)
        win._enable_run_buttons()

    if shot_path:                                # 自检：定时截图后退出
        def _grab():
            pix = win.grab()
            pix.save(shot_path)
            print(f"[shot] saved {shot_path} {pix.width()}x{pix.height()}")
            print(f"[dbg] stat: {win.stat.text()}")
            print(f"[dbg] meta: {win.meta.text()}")
            print(f"[dbg] banner: {win.banner.text()}")
            print(f"[dbg] banner_pending: {len(win.banner_pending)} "
                  f"hist: {win.hist_lab.text()}")
            print(f"[dbg] imgNone: {win.video._img is None} "
                  f"queue_rows: {sum(1 for k in range(win.queue_lay.count()) if win.queue_lay.itemAt(k).widget())}")
            print(f"[dbg] spots: {win.dispatch.spot_state}")
            win.write_rec_meta()
            win.worker.stop()
            win.worker.wait(1500)
            win.voice.shutdown()      # 杀掉播放中/排队的 paplay，避免占用声卡
            app.quit()
        QTimer.singleShot(shot_ms, _grab)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
