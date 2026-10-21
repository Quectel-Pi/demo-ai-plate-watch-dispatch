# 智能车牌识别调度系统

[English](README.md) | 简体中文

智能车牌识别调度系统是基于 Quectel Pi（RK3576）的边缘 AI 车牌识别与接客调度应用。播放真实道路视频或接入 USB 摄像头，在板载 NPU 上实时完成车牌检测、四角点几何校正、文字识别与多帧确认，并自动驱动「登记用车 → 四车位分配 → 到位接客 → 完成/超时/取消 → 候队补位」的全流程调度，关键动作全程语音播报——全程离线运行，画面与数据不离开设备。

![界面](assets/app-preview.jpg)

## 功能特性

- **实时车牌识别** — HyperLPR3 双模型（检测 y5fu + 识别 rpv3，RKNN fp16）跑在 RK3576 NPU 上，720p 视频实测 14~18 FPS
- **蓝/绿牌通吃** — 普通蓝牌 7 位、新能源绿牌 8 位，含省份汉字
- **多帧确认去重** — IoU 轨迹关联 + 同一读数 ≥3 票/占比/置信度/长度四重门闸，杜绝截断读数与误报
- **车位自动调度** — 识别到已登记车牌自动分配 A01~A04 车位，横幅大字 + 语音播报「车牌号，请前往 A0X」
- **到位与接客管理** — 车辆到位、接客完成、超时警示、订单取消、车位释放一气呵成
- **候车队列 FIFO 补位** — 车位满自动排队，先到先得，一有空位依序补位
- **未登记提醒与补登记** — 视频里识别到未登记车辆自动发现并提示，支持一键补登记
- **二次接客** — 已登记车辆再次到来无需重复登记，直接安排接客
- **中英双语** — 界面与语音一键切换，离线语音包随包内置
- **一键预演模式** — 点「预演模式」自动演示调度全流程（到位/完成/超时/取消/补登记/二次接客）
- **记录导出** — 识别事件（JSON/CSV）+ 车牌截图 + 调度流水日志，会话结束自动落盘

## 技术栈

- Python 3 + PySide6（Qt）界面，深色科技风 QSS + 等宽字体数据读出
- HyperLPR3 模型（y5fu 检测 / rpv3 识别）转换为 RKNN（fp16 不量化），ctypes 直调 librknnrt 在 NPU 推理
- OpenCV 视频文件 / V4L2 摄像头采集，角点实测宽高透视校正
- edge-tts 生成的离线语音词元库（省份/字母/数字/短语按句拼接），PipeWire 播放、ALSA 兜底
- Go 静态启动器 + `.run` 自解压安装包

## 硬件要求

| 项目 | 要求 |
|---|---|
| 板子 | Quectel Pi（RK3576，6 TOPS NPU） |
| 摄像头（可选） | USB UVC 摄像头（识别模式也可播放内置演示视频） |
| 音频 | 扬声器或耳机（跟随系统输出路由，支持蓝牙耳机） |
| 存储 | 安装约需 1 GB 空间 |

## 项目结构

```
├── code/                    # 应用源码
│   ├── main.py              # 入口
│   ├── app.py               # 界面、视频循环、调度事件渲染
│   ├── detector.py          # 车牌检测（y5fu 后处理）
│   ├── reader.py            # 文字识别（rpv3 CTC 解码）
│   ├── tracker.py           # 轨迹关联与多帧确认门闸
│   ├── dispatch.py          # 调度核心（分配/到位/超时/候队/预演）
│   ├── voice_player.py      # 语音词元拼接与三级播放回退
│   ├── session.py            # 会话记录与导出
│   ├── rknn_backend.py      # RKNN NPU 推理封装（ctypes）
│   ├── run.sh               # 启动脚本（Wayland 会话发现）
│   └── plate-recognition    # 双击启动器（Go 静态 ELF）
├── models/                  # 检测/识别模型 ONNX + RKNN
├── assets/voice/            # 离线语音词元库（80 条 wav）
├── materials/videos/        # 5 段开箱演示视频
├── materials/demo/          # 三段式应用演示视频
└── tools/                   # 模型转换、语音生成、打包脚本
```

## 快速开始

### 1. 获取安装包

从 [Releases](https://github.com/Quectel-Pi/demo-ai-plate-watch-dispatch/releases) 页面下载 `plate-recognition-installer.run`（约 137 MB，内置 Python 运行时、模型、语音包与演示视频）。

### 2. 推送到板子

```bash
adb push plate-recognition-installer.run /home/q/
adb shell chown q:q /home/q/plate-recognition-installer.run
```

### 3. 安装

```bash
# 普通用户 q（推荐）→ ~/plate-rec
adb shell su - q -c '/home/q/plate-recognition-installer.run'

# 或 root → /root/plate-rec
adb shell /home/q/plate-recognition-installer.run
```

### 4. 启动

在文件管理器中双击 `code/plate-recognition`，或：

```bash
cd ~/plate-rec/code && ./run.sh
```

## 从源码构建（开发者可选）

> 普通用户直接下载 [Releases](https://github.com/Quectel-Pi/demo-ai-plate-watch-dispatch/releases) 的 `.run` 安装包即可，无需走本节。
> 打包在**宿主机（x86 Linux）**上进行；开发板只负责安装运行。

### 1. 克隆仓库并检查前置条件

```bash
git clone https://github.com/Quectel-Pi/demo-ai-plate-watch-dispatch.git
cd demo-ai-plate-watch-dispatch

go version          # Go ≥ 1.18（编译双击启动器；没有则复用仓库已带的 ELF）
python3 --version && tar --version && xz --version
adb devices         # 已连上开发板
```

### 2. 构建 Qt 运行时（runtime/，仓库不含此目录）

界面依赖 PySide6/Qt6（aarch64、Python 3.13）。用 micromamba 拉一套 aarch64 环境再按目录归位：

```bash
micromamba create -y -p ./rt-root -c conda-forge --platform linux-aarch64 \
    python=3.13 pyside6=6.11.2

mkdir -p runtime/py runtime/lib runtime/share
cp -a rt-root/lib/python3.13/site-packages/. runtime/py/          # PySide6 + shiboken6
( cd rt-root/lib && find . -maxdepth 1 -mindepth 1 ! -name 'python*' \
    -exec cp -a {} "$OLDPWD/runtime/lib/" \; )                    # Qt 库与依赖 .so
cp -a rt-root/share/. runtime/share/                             # Qt 翻译等
```

组装出的 `runtime/` 即可用（未裁剪约 1GB）；进一步用 readelf NEEDED 闭包裁到 151MB 的方法见 [TECH.md](TECH.md) 9.3 节。已在板上装过本应用时，也可直接复用安装目录里的 `runtime/`。

### 3. （可选）语音与模型再生成

仓库已内置全部产物，以下仅在需要重制时执行：

```bash
# 语音词元（assets/voice/，80 条 wav）：需 edge-tts + miniaudio
pip install edge-tts miniaudio
python3 tools/gen_voice.py

# 模型（models/，ONNX 已在仓库内）：需 rknn-toolkit2 2.3.2（opset19、fp16、target rk3576）
# 转换细节见 TECH.md 第 10 节
```

### 4. 一键打包

```bash
bash tools/make_run_pack.sh
# 产物：dist/plate-recognition-installer.run（约 137MB）
```

脚本自动完成：Go 交叉编译启动器 → 组装 code/models/runtime/演示视频/语音 → tar.xz 压缩 → 拼 shell 自解压头。

### 5. 部署到板子

```bash
adb push dist/plate-recognition-installer.run /home/q/
adb shell chown q:q /home/q/plate-recognition-installer.run
adb shell su - q -c '/home/q/plate-recognition-installer.run'
```

安装完成后双击 `plate-rec/code/plate-recognition` 启动，验收清单与日常迭代见 [快速开始](#快速开始)。

> **日常改代码**：无需重新打包，直接把改动的 `code/*.py` 推到板上安装目录
> `~/plate-rec/code/` 即生效（`adb push code/app.py /home/q/plate-rec/code/`）。

## 使用方法

1. 点「登记乘客用车」输入车牌，或点「预置演示车辆」批量登记
2. 点「打开视频」播放 `materials/videos/` 里的视频（或接 USB 摄像头点「摄像头识别」）
3. 识别到已登记车牌 → 自动分配车位，横幅 + 语音「车牌号，请前往 A0X」
4. 点该车辆行内「车辆到位」→ 开始接客；点「接客完成」→ 释放车位，候队自动补位
5. 超时车辆车位卡红色警示，可取消释放；顶栏实时显示完成/等待/车位占用统计

**零手动替代**：点队列区「预演模式」（或 `--demo` 参数）自动演示到位/完成/超时/取消/补登记/二次接客全流程。

### 语音播报

- 登记分配：「苏A53D9P，请前往 A01」
- 到位接客：「苏A53D9P，已到位 A01，开始接客」
- 接客完成 / 订单取消 / 超时警示均有对应播报，横幅与语音严格同步

## 常见问题

**检测不到摄像头？**

板上执行 `ls /dev/video*` 确认设备，重新插拔 USB 摄像头；也可先用 `materials/videos/` 演示视频体验。

**识别不准？**

半可见车牌只统计完整读数（≥7 位）后才确认事件；逆光、剧烈模糊会降低识别率，正面角度效果最佳。

**没有语音？**

应用跟随系统音频输出路由。检查系统设置的输出设备，蓝牙耳机同样可用。

## 技术说明

模型转换（ONNX → RKNN）、识别门闸参数、调度事件流与完整踩坑记录，详见 [TECH.md](TECH.md)。
