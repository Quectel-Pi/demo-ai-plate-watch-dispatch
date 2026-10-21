# 智能车牌识别调度系统 · 技术资料

> 本文档是项目的技术参考手册，面向开发/维护者。部署与使用见 [README.zh-CN.md](README.zh-CN.md)。
> 硬件平台：Quectel Pi（RK3576，aarch64，NPU 6 TOPS）· 全程离线，无云端依赖。

---

## 1. 系统概览

### 1.1 数据流

```
视频文件 / USB 摄像头
   │  OpenCV 采集（摄像头：V4L2 超时 + BUFFERSIZE=1 低延迟）
   ▼
y5fu 车牌检测（RKNN fp16, 640×640, NPU）
   │  输出 [n,14]：x1,y1,x2,y2,score + 4 角点(8) + 单双层分类
   ▼
按角点实测宽高透视校正（保留真实宽高比）
   ▼
rpv3 文字识别（RKNN fp16, 48×160, NPU）→ CTC 解码（78 字符表）
   ▼
Tracker：IoU 轨迹关联 + 多帧投票 + 四重门闸确认
   │
   ├─► PySide6 实时三状态标注（检测中 / 候选中 / 已确认）
   ├─► Session：事件冻结 + 车牌截图 + JSON/CSV 导出
   └─► DispatchCore 调度核心：确认车牌 → 登记匹配 → 四车位分配
         → 事件流（分配/到位/完成/取消/超时/未登记/等待）
         → 三区界面渲染 + 关键动作语音播报 + dispatch_log 落盘
```

### 1.2 线程模型

- **推理全部在 QThread（`Worker`）内**：OpenCV 采集、NPU 推理、跟踪、会话记录都不占 UI 线程；
  UI 线程只做渲染与交互。
- **信号回传**：`frameReady(QImage, snaps, stats)`（每帧）、`eventConfirmed(dict)`（确认事件）、
  `videoFinished(str, str)`、`exportDone(str)`、`camFailed()`、`logMsg(str)`。
- **跨线程 numpy 必须独立副本**（`.copy()`）：槽是异步执行的，原 dict 会被
  `Session.add_event` 先 `pop('best_plate')`，不拷贝则事件卡片缩略图取到 None 变黑。
- 已确认轨迹**跳过 NPU 识别**（事件一次性冻结），实测帧率 12.4 → 13.9 FPS。
- 视频模式自适应跳帧：`step = max(1, ceil(ema_proc * fps))`，按处理耗时平滑跟帧。
- **暂停即冻结**：`paused` 时 `msleep(50)+continue` 跳过 `cap.read()`，文件指针不动，
  后续帧不会被识别；轨迹超时按视频内时间 `frame_idx/fps` 计，暂停多久都不丢候选轨迹。

---

## 2. 输入源

### 2.1 视频模式

- 支持 mp4/mov/avi/mkv；启动参数可直接带路径（相对路径按 cwd → APP_ROOT 依次解析，
  因 `run.sh` 会先 `cd code/`）。
- 命令行：`run.sh [video] [--lang zh|en]`。

### 2.2 摄像头模式（关键：V4L2 发现策略）

板上 `/dev/video-camera0`（→ video51）是 **rkisp_mainpath 多平面（mplane）节点**：
ISP media 管道未配置时 `open()` 成功但 `VIDIOC_STREAMON` 永远返回 Invalid argument
（dmesg：`nonsupport pixelformat:YUYV` / `check rkisp_mainpath link or isp input`）。
真实可用的是 USB（UVC）摄像头（板上实测索引 73）。

`find_cameras()` 三重过滤（`code/app.py`）：

1. **V4L2 QUERYCAP ioctl**：只保留带 `V4L2_CAP_VIDEO_CAPTURE`（bit 0）的**单平面**节点
   —— mplane 节点无此位，自动排除（`_v4l2_probe`）；
2. **USB 总线优先排序**（`bus_info` 以 `usb` 开头优先）；
3. **open + read 实测出帧**才收录（打开带 1500ms 超时）。

运行参数：open/read 超时 2000ms、`CAP_PROP_BUFFERSIZE=1`、
分辨率协商 1280×720 → 960×540 → 640×480、连续 30 次 read 失败判定丢失自动归档会话。
会话名 `camera_%Y%m%d_%H%M%S`，统计栏不显示总帧数（实时模式 `total=0` 区分）。
命令行：`run.sh --camera [dev]`（dev 可为索引或 /dev/videoN 路径）。

---

## 3. 车牌检测（detector.py）

- **模型**：HyperLPR3 自训 YOLOv5n 多任务变体 `y5fu`（640 输入），
  同时回归车牌框、4 角点、单/双层牌分类，转出 `hlpr_det640_fp16.rknn`。
- **预处理**：letterbox 等比缩放到 640×640 画布居中补零，**0-1 归一化**
  （ONNX 即如此，RKNN 转换用 std=255）。
- **解码**：输出 `[n,14]` = `x1,y1,x2,y2,score,kp8(x,y×4),layer`，映射回原图坐标；
  置信阈值 0.3，NMS IoU 0.5。
- 实测：检测耗时 15~55ms（640），另有 `hlpr_det320_fp16.rknn` 可换（更快、小牌稍弱）。

## 4. 文字识别（reader.py）

- **模型**：HyperLPR3 `rpv3`（CRNN+CTC，输入 48×160，78 字符表含省份汉字/港澳警学挂），
  转出 `rpv3_fp16_rk3576.rknn`。**归一化 (x-127.5)/127.5**（与检测不同，转换时固化 mean/std）。
- **预处理必须复刻官方两步**（缺一不可）：
  1. `get_rotate_crop_image`：按 4 角点**实测宽高比**透视校正（蓝牌 ~3.1 / 绿牌 ~3.4），
     保留原始分辨率与真实宽高比——**严禁固定 94×24 直拉**（会横向畸变字符）；
  2. `encode_images`：等比缩放到高 48；宽 `ceil(48*ratio) > 160` 时直接拉到 160，
     否则缩放后右侧补零到 160。
- **CTC 解码**：blank=0，按绿牌 8 位/蓝牌 7 位输出完整读数。

## 5. 跟踪与多帧确认（tracker.py）

### 5.1 轨迹生命周期

| 阶段 | 规则 |
|---|---|
| 关联 | 帧间 IoU ≥ 0.2 匹配到已有轨迹，否则开新轨迹（track_id 递增） |
| 存活 | 距上次更新 ≤ 2.5s 保持，超时未确认则归入 `unresolved` |
| 确认 | 达成四重门闸即冻结（`confirmed=True`），**此后跳过 NPU 识别** |
| 质量 | 每帧算质量分 `q = det_conf × 角点框高`，最佳帧截图存 `best_plate` |

### 5.2 确认四重门闸（全部满足才导出）

| 门闸 | 阈值 | 目的 |
|---|---|---|
| 票数 | 同一读数 ≥ **3** 帧 | 多帧一致 |
| 占比 | 读数票数 / 有效读数总数 ≥ **0.5** | 排除偶然对 |
| 置信 | 读数平均字符置信度 ≥ **0.80** | 拦远距低置信垃圾读数 |
| 长度 | 读数 ≥ **7** 字符才参与表决 | 半可见车牌 0.99 置信的**截断读数**（如「苏A」）抢跑 |

另有**质量门闸**：车牌像素高 < **14px** 的帧不参与表决（计 `low_quality_reads`）。
四重门闸的由来：实测「半可见车牌产生 0.99 置信截断读数凑 3 票抢跑确认」是确认错误的
主因（1.1s 用远距帧确认「苏AQ2」后，正确读数「苏AQ2L95」不再导出）——
长度门闸上线后 lpr_test 5/5 全对。

### 5.3 状态机（UI 三状态的来源）

`detected`（灰）→ `candidate`（橙，带 x/need 票数）→ `confirmed`（绿，读数冻结）。

---

## 6. 会话与记录（session.py）

- 每次「重新会话」/视频播完/摄像头丢失时归档：`data/<时间戳>/`
  （`os.makedirs(..., exist_ok=True)` 自动建目录，`PLATE_REC_DATA` 环境变量可重定向）。
- `events.json`：`session_id/video/started_at/events[]/unresolved[]`；
  事件字段 = `track_id, plate, plate_type(blue|green, 按读数长度 ≥8 判绿), first_seen_s,
  confirmed_s, frames, char_conf, best_q`。
- `events.csv`：同内容 utf-8-sig（Excel 直开不乱码）。
- `crops/<牌号>_<轨迹号>.jpg`：确认时最佳帧车牌 4 倍放大截图。
- `unresolved`：超时轨迹的全部候选读数，供困难样本排查。

---

## 7. 图形界面（app.py，PySide6）

### 7.1 深色科技风实现（三件套缺一不可）

1. `app.setStyle('Fusion')` —— 跨平台控件行为一致；
2. **深色 QPalette**（`build_dark_palette()`）—— 兜住 QSS 覆盖不到的原生控件
   （编辑框光标/下拉弹层/tooltip），保证无浅色残留；
3. **全局 QSS** —— 暗底 `#0b1220`、面板 `#0e1729`、青绿 `#22d3ee` 高亮、琥珀 `#fbbf24` 候选。

视觉要点：

- 画面边缘**三层青色光晕描边**（13px α22 / 7px α45 / 3px α90 由外向内渐亮）；
- 检测框：检测中=雷达虚线 / 候选=琥珀实线+**瞄准角括号** / 已确认=发光角括号（锁定）；
- 待机画面四角暗青括号（等待信号）；
- 数据读出区（FPS/票数/牌号）**等宽字体**：板上 `DejaVu Sans Mono` 可用，
  汉字由 fontconfig 回退 CJK 字体；
- 主按钮青色渐变 + 亮边；状态点 `qradialgradient` 径向光晕。

### 7.2 文件对话框深色化（踩坑）

原生 GTK 对话框**不响应 QSS** → 深底黑字不可读。修复：
`QFileDialog.setOption(DontUseNativeDialog, True)` 强制 Qt 渲染，
并在 QSS 中给 `QFileDialog QListView/QTreeView/QLineEdit/QComboBox` 显式配色。
QMessageBox 统一走 `_info_box()` 辅助函数（parent 主窗 + 按钮文案随语言）。

### 7.3 i18n 中英切换

模块级 `TEXTS['zh'|'en']` 双语字典 + `tr(key, **kw)`；`set_lang()` 切换后
`retranslate()` 全量刷新静态控件文案。启动 `--lang en` 或 `PLATE_LANG=en` 指定初始语言。

---

## 8. 接客调度终端（dispatch.py + voice_player.py）

识别内核之上的业务层：登记乘客用车 → 识别匹配 → 四车位（A01~A04）自动分配 →
到位 → 接客完成。**纯逻辑无 UI 依赖**，宿主机可直接单测驱动。

### 8.1 状态机

```
订单: waiting(等车位) → assigned(已分配) → serving(接客中) → done(完成)
      活跃态任意时刻可 cancel
车位: free → assigned → serving → free
```

车位只在 `_assign()` 一处变更，调用方仅两个：`register()`（登记即分配）与
`_refill()`（释放后 FIFO 补位）——**登记是进车位的唯一入口**，未登记车辆
永远碰不到车位。

### 8.2 调度规则与参数

| 规则 | 参数 | 说明 |
|---|---|---|
| 登记即分配 | — | 有空位直接分；无空位进 FIFO 等待队列 |
| 重复识别保持原车位 | `REANNOUNCE_S=15` | assigned 态再识别 >15s 才重播提醒 |
| 已到位识别静默 | — | serving 态重播「请前往」语义错乱，不出事件 |
| 等待中识别节流 | `WAITING_REMIND_S=60` | 等待是常态，仅刷横幅/日志 |
| 未登记提示冷却 | `UNREG_COOLDOWN_S=60` | 按车牌独立计时，支持现场补登记 |
| 分配超时 | `TIMEOUT_S=120`（环境变量 `DISPATCH_TIMEOUT_S`） | 车位保留，可到位/取消 |

### 8.3 事件流与播报治理

`on_plate_seen(plate, t)` 等方法返回事件 dict 列表，UI 层 `_apply_events()`
分发到四处：横幅、历史条、语音、日志（`dispatch_log.jsonl` 全事件 +
会话终态 `dispatch_snapshot_*.json`）。

**语音原则：只播需要司机立即行动的时刻**（分配/到位/完成/取消/超时）；
未登记与等待中仅视觉提示——治理前 8 类事件全播，播视频时「4 条分配连发 +
等待车反复播 + 未登记车乱入」形成轰炸，治理后实测（rk_5s 预置 9 辆）全程仅
welcome + 4 条「车牌，请前往 A0x」。

**横幅与语音同步机制**（解决「横幅已切到第 4 条、语音还在播第 1 条」）：
每条语音播报携带自己的横幅文案闭包，voice 线程**开播即回调** →
`Signal(str, bool)` 队列到 UI 线程切横幅——横幅永远显示正在播的这条，
事件积压也不会错位（闭包自带文案，不依赖队列计数配对）。
横幅下方**历史条**（小字等宽，最近 3 条带时间戳）记录全部事件——
「完成 + 补位」这类同一瞬间成对出现的事件在历史条一目了然。
无语音事件（未登记/等待/重复拒绝）仅在横幅空闲时直显，始终进历史条。
语音关闭（`PLATE_VOICE=0`）时横幅队列由 `_tick` 定时推进（2.8s/条）。

### 8.4 语音链路（voice_player.py）

- 字符级 WAV 拼接（`assets/voice/`：省份/字母/数字/短语），`GAP_S=0.20`
  token 间静音（0.12s 实测听感急促）；
- 播报顺序 = 真实调度广播惯例：**车牌在前、指令在后**——「苏A53D9P，请前往
  A01」、「苏A53D9P，已到位 A01，开始接客」（到位语序：车牌 + 已到位 +
  车位号 + 开始接客，`arrived_verb`/`serve_start` 两个短语词元拆分拼接，
  避免车位号掉到句尾的歧义）；
- 板上音频三级回退：
  1. 本用户直连 `paplay`（走 PipeWire，跟随系统输出设备，蓝牙耳机可用）；
  2. root 会话 `su - <会话用户> -c 'XDG_RUNTIME_DIR=… paplay …'`——
     Pulse 协议拒绝 root 直连非 root 会话，且 q 的 PipeWire 独占声卡令
     root 的 aplay 报「设备忙」，**su 是 root 会话唯一活路**；
  3. `aplay -D plughw:0,0` ALSA 硬件兜底；
- `PLATE_VOICE=0` 整体关闭；`--shot` 自检退出必须显式 `voice.shutdown()`
  （`app.quit()` 不走 closeEvent，残留 paplay 会占声卡）。

### 8.5 预演模式（--demo / 界面「预演模式」按钮，全自动演示调度全流程）

`./run.sh --demo [视频]` 或点队列区「预演模式」按钮（同一入口 `start_demo`，
自带防重入：预演进行中再点忽略）。预演换 30s 短超时核心（真实默认 120s
等太久），登记按 `DEMO_SCHEDULE` 计划表在 `_demo_tick`（1s tick）里错峰触发
——一次性 9 辆会让「请前往」播报积压到音画不同步，分批后每条播报紧贴画面
事件；各时序可用 `PLATE_DEMO_ARRIVE/SERVE/REG/TIMEOUT/CANCEL_S` 环境变量
压缩（演示视频录制用，默认保持产品节奏）。零手动跑通全部调度分支：

| 时序（真实时钟） | 演示的情况 |
|---|---|
| 8s / 9s | 苏A53D9P → A01、苏AQ2L95 → A02（登记分配 +「请前往」播报） |
| 24s | 苏ABD220 → A03、苏A0AR19 → A04（后者故意不到位，专门触发超时分支） |
| 25s | 苏A53D9P 重复登记 → 拒绝（只上横幅，不弹模态框打断预演） |
| 26s | 皖ADJ6382 登记 → 四车位已满 → FIFO 候队 |
| 分配后 8s / 到位后 10s | 车辆到位（车位卡变绿）→ 接客完成释放 → 队列头同批 FIFO 补位 |
| 超时窗口 30s 到点 | 苏A0AR19 超时红显 → 5s 后自动取消释放（「取消释放+补位」分支） |
| 识别到未登记车牌 3s | 自动补登记（视频里的车真实触发，走正常补登记入口） |
| 全部流转完 | 苏A53D9P 二次接客，再走一遍完整流程 |

`--record` 配合预演录制演示视频素材：10fps 帧序列 + `meta.json` 事件时间线 +
语音拼接 wav 留存，宿主机按时间线混音合成成片（`materials/demo/
plate-recognition-demo.mp4`，三段式 191.8s：项目介绍 → 功能介绍 → 板上整体预演）。

---

## 9. 启动器与打包

### 9.1 双击入口（tools/launcher/main.go）

Go 静态交叉编译（`GOOS=linux GOARCH=arm64 CGO_ENABLED=0`）的 ELF，
GNOME 双击直接执行（无「运行/显示」弹框，AppImage 原理）。职责：

1. `/proc/self/exe` 上两级定位应用根；
2. `pkill -f "[a]pp.py"` 杀旧实例（NPU 单实例）；
3. 日志追加到 `~/.local/state/plate-rec/app.log`；
4. `setsid` 脱离进程组调 `run.sh`，**命令行参数原样转发**（`os.Args[1:]`，
   `plate-recognition --camera` 全靠它——曾踩坑参数被吞）。

### 9.2 run.sh 会话发现

优先使用调用者自身的 Wayland 环境（双击场景天然有）；adb root 等无会话环境才扫描
`/run/user/*/wayland-*` 借普通用户会话，都无则回退 xcb。

### 9.3 自解压 .run（tools/make_run_pack.sh，一键打包）

```bash
bash tools/make_run_pack.sh   # 自动编译启动器(有 Go) + 组装 + 打包
```

- 内部：① 检测 Go 编译启动器（无 Go 复用已有 ELF，均无则报错）→ ② 暂存组装
  （python shutil，规避宿主 cp/rm 安全拦截）→ ③ tar.xz 压缩 → ④ shell 头 + `sed`
  回填 payload 起始行 → ⑤ 拼成单个 `plate-recognition-installer.run`（约 137MB）。
- 板上解压用 **python3 内置 lzma**（`tarfile mode="r|xz"`）—— 板上最小系统无 `xz` 命令；
  payload 必须用 `python3 -c` 传参，heredoc 会抢占 stdin 导致管道读不到数据。
- 安装器逻辑：默认 `~/plate-rec`，检测旧版清理重建，拒绝装 `/` 或 `$HOME` 根，
  root 装到 `/home/*` 自动还属主。
- **Qt runtime 裁剪**：readelf（可跨架构读 aarch64 ELF）解析 PySide6 Core/Gui/Widgets
  + wayland/xcb 插件的 NEEDED 闭包，985MB → 151MB。
  坑：NEEDED 指向 `.so.6` 符号链接，必须连同 realpath 真实文件一起拷。

---

## 10. 模型转换（宿主机）

- 工具链：**rknn-toolkit2 2.3.2 + opset19 + fp16 不量化 + target_platform=rk3576**
  （转换脚本：`tools/convert_det_rknn.py` / `fix_rpv3_onnx.py` / `fix_lprnet_onnx.py`）。
- **rpv3 原始 ONNX 有 101 个输入**（100 个是「输入+initializer 二重身」权重），
  需 ONNX 手术清除后才能转 RKNN；LPRNet 的 Einsum 陷阱用
  ReduceMean→AveragePool+Reshape 替换解决。
- 归一化差异：y5fu 是 0-1（转换 std=255），rpv3/LPRNet 是 (x-127.5)/127.5
  （转换时固化 mean/std）；宿主机对照测试必须先 transpose NCHW 再对应归一化。
- fp16 实测检测模型角点噪声无影响（蓝牌角点宽高比 2.9~3.2 健康）。

---

## 11. 性能实测（板上）

| 场景 | 帧率 | 检测耗时 | 备注 |
|---|---|---|---|
| lpr_test.mp4（1080p） | 9.8~12 FPS | 55ms | 5 蓝牌 |
| rk 系列（720p） | 9.5~18 FPS | 47~51ms | 分析帧率 |
| hw_day.mp4（1080p 高速） | 14.6 FPS | — | C98191P |
| USB 摄像头实时 | 13.3~18.2 FPS | 47~54ms | 索引 73 |
| 模型加载 | 0.06~0.08s | — | 双 RKNN fp16 |

## 12. 验证结果（10 确认事件，全部与人工真值一致）

| 视频 | 场景 | 确认事件 |
|---|---|---|
| lpr_test.mp4 | 城市道路 | 苏A53D9P / 苏AQ2L95 / 苏ABD220 / 苏A0AR19 / 苏A3XF72 |
| rk_2s.mp4 | 城市道路 | 皖ADJ6382（绿） |
| rk_5s.mp4 | 城市道路 | 皖AA91218（绿）+ 京AD79091/京ADZ9091（第二辆低清，人工核对倾向京ADZ9091） |
| rk_9s.mp4 | 城市道路 | 皖AA88515（绿） |
| hw_day.mp4 | 高速监控 | C98191P（海外样式，Mendeley HVSD CC BY 4.0；night/rain 段过暗/低清不可用） |

真值登记：`materials/annotation/ground_truth.json`；验证产物：`materials/verification/`。

调度层验证：板上 `--preset` + rk_5s 实测——4 车位分配、
FIFO 候车、未登记零车位、语音仅 welcome + 4 条分配播报、
`dispatch_log.jsonl` / `dispatch_snapshot_*.json` 落盘齐全。

## 13. 踩坑速查

| 症状 | 根因 | 解法 |
|---|---|---|
| 摄像头「未找到」 | rkisp mplane 节点 open 成功但 STREAMON 失败 | QUERYCAP 单平面过滤 + USB 优先 + 实测出帧 |
| 半可见车牌确认错误读数 | 0.99 置信截断读数凑票抢跑 | 读数长度 ≥7 才参与表决 |
| 事件卡片缩略图黑 | 跨线程槽异步执行，numpy 载荷被 pop | Signal 传参用 `.copy()` 独立副本 |
| 文件对话框深底黑字 | 原生 GTK 对话框不响应 QSS | `DontUseNativeDialog` + QSS 显式配色 |
| `--camera` 双击失效 | launcher 不转发参数 | `exec.Command(run.sh, os.Args[1:]...)` |
| .run 安装卡死/读不到 payload | 板上无 xz；heredoc 抢占 stdin | python3 lzma + `python3 -c` 传参 |
| 中文标注乱码 | cv2.putText 不支持中文 | QPainter 叠加绘制 |
| 识别帧率低 | 已确认轨迹仍跑识别 | 确认即冻结，跳过 NPU 识别 |
| 板上时钟慢半年 | 硬件时钟未同步 | 对照宿主机日志时注意会话目录名按板钟 |

---

## 14. 目录结构

```
AI-Plate-Watch-Inside/
├── code/                 # 板端运行代码
│   ├── app.py            # PySide6 界面（Worker/VideoWidget/EventCard/MainWindow）
│   ├── detector.py       # y5fu 检测（letterbox + 解码 + NMS）
│   ├── reader.py         # rpv3 识别（透视校正 + 48×160 encode + CTC）
│   ├── tracker.py        # IoU 轨迹关联 + 四重门闸多帧确认
│   ├── session.py        # 会话记录（events.json/csv + crops）
│   ├── dispatch.py       # 接客调度核心（纯逻辑，无 UI 依赖）
│   ├── voice_player.py   # 调度语音（WAV 拼接 + 三级音频回退）
│   ├── rknn_backend.py   # ctypes 直调 librknnrt.so
│   ├── main.py           # 无界面 CLI 测试入口（python3 main.py <video>）
│   └── run.sh            # 终端启动入口
├── models/               # .rknn（fp16）与原始 ONNX
├── materials/
│   ├── videos/           # 5 段演示视频
│   ├── annotation/       # ground_truth.json
│   ├── verification/     # M2 验证产物
│   ├── crops_for_review/ # 待人工核对截图
│   └── demo/             # 应用演示视频成片（--record 素材在临时目录合成）
├── runtime/              # 裁剪 PySide6/Qt6 aarch64 运行时（lib/ + py/）
├── tools/
│   ├── launcher/         # Go 启动器（main.go + build_launcher.sh）
│   ├── make_run_pack.sh  # 一键打包脚本
│   ├── board/            # 板上基准/联调脚本
│   └── *.py              # 模型转换/宿主机扫描/真值裁决等工具脚本
├── assets/
│   ├── voice/            # 拼接语音 WAV（省份/字母/数字/短语/车位名）
│   └── app-preview.jpg   # README 界面预览图
├── materials/
│   ├── videos/           # 5 段开箱演示视频
│   └── demo/             # 三段式应用演示视频成片
├── .build/               # 宿主机构建环境（rknn-host 转换环境 / ultra-env）
├── dist/                 # plate-recognition-installer.run 产物（本地构建，不入库）
└── docs/                 # 开发期界面截图与文档（仓库内技术资料见 TECH.md）
```

> 源码树中曾有 `config/`、`data/` 两个空目录：config 无代码引用、data 由
> `Session` 运行时 `makedirs` 自动重建，均不需要在源码树占位。

---

## 15. 后续规划

- **车位占用检测**（第二阶段）：固定机位、单独模型/逻辑；
- M3 预案（自训 yolo11n-pose 关键点检测）暂不启动——当前检测 9/9 全对。
