#!/bin/bash
# 交叉编译板上双击启动器（静态 aarch64 ELF）
# 产物: tools/launcher/plate-recognition  ->  打包时被 make_run_pack.sh 收进 code/
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o plate-recognition .
echo "==> 完成: $DIR/plate-recognition"
file plate-recognition
