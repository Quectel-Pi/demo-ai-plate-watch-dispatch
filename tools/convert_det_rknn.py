#!/usr/bin/env python3
"""主机端：车牌四角点检测 YOLOv8m-pose ONNX → RK3576 fp16 RKNN（不量化保精度）。

用法: python3 tools/convert_det_rknn.py <in.onnx> <out.rknn>
依赖: 复用健身项目验证过的 rknn-toolkit2 2.3.2 隔离环境（只读 PYTHONPATH）。
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/q/project/Demo/AI-Fitness-Coach/.build/rknn-env")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    # Ultralytics 归一化: x/255 -> mean=0, std=255；NHWC 由 runtime 处理
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3576",
    )
    if rknn.load_onnx(model=str(src)) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:  # fp16 不量化
        raise SystemExit("build failed")
    if rknn.export_rknn(str(dst)) != 0:
        raise SystemExit("export failed")
    rknn.release()
    print(f"[ok] {dst} ({dst.stat().st_size // 1048576} MB)")


if __name__ == "__main__":
    main()
