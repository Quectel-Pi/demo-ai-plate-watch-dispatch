#!/bin/bash
# 智能车牌识别调度系统 板上启动脚本：自动发现 Wayland 桌面会话，不写死显示节点
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"

export LD_LIBRARY_PATH="$DIR/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DIR/code:$DIR/runtime/py"
export QT_PLUGIN_PATH="$DIR/runtime/lib/qt6/plugins"

# 识别记录/截图默认写应用目录 data/（安装目录属主即运行用户，可写）；
# 也可用 PLATE_REC_DATA 覆盖
export PLATE_REC_DATA="${PLATE_REC_DATA:-$DIR/data}"

# ---- 桌面会话定位 ----
# 普通用户双击: 自己的会话环境天然存在；adb root 等无会话环境才扫描借用
if [ -z "${WAYLAND_DISPLAY:-}" ] || [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    for udir in /run/user/*; do
        [ -d "$udir" ] || continue
        for wl in "$udir"/wayland-*; do
            [ -S "$wl" ] || continue
            export XDG_RUNTIME_DIR="$udir"
            export WAYLAND_DISPLAY="${wl#"$udir"/}"
            export DBUS_SESSION_BUS_ADDRESS="unix:path=$udir/bus"
            export XDG_SESSION_TYPE=wayland
            export QT_QPA_PLATFORM=wayland
            break 2
        done
    done
    [ -n "${WAYLAND_DISPLAY:-}" ] || export QT_QPA_PLATFORM=xcb
else
    export QT_QPA_PLATFORM=wayland
fi

export FONTCONFIG_FILE="/etc/fonts/fonts.conf"

cd "$DIR/code"
exec python3 -u app.py "$@"
