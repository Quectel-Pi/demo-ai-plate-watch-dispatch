#!/usr/bin/env python3
"""板上 NPU 推理基准：预热 + 多次迭代，输出稳态耗时。

用法: python3 bench_det.py <model.rknn> [size] [iters]
"""
import sys
import time

import numpy as np

from rknn_backend import RKNNRuntime


def bench(path, size=640, iters=20):
    rt = RKNNRuntime(path)
    x = (np.random.rand(1, size, size, 3) * 255).astype(np.float32)
    for _ in range(3):                      # 预热（含冷启动）
        rt.run(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        rt.run(x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts)
    med = ts[len(ts) // 2]
    print(f"{path.split('/')[-1]} @{size}: 中位={med:.1f}ms "
          f"min={ts[0]:.1f} max={ts[-1]:.1f} → {1000/med:.1f} FPS")
    rt.close()


if __name__ == "__main__":
    model = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    bench(model, size, iters)
