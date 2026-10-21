#!/usr/bin/env python3
"""修复 lprnet.onnx 的 ReduceMean_62（axes=[2]）：
toolkit2 2.3.2 会把它图优化成 Einsum，NPU/runtime 均不支持导致板上崩溃。

等价替换：ReduceMean(axes=[2], keepdims=0) == AveragePool(kernel=[4,1]) + Reshape
AveragePool/Reshape 均为 NPU 原生支持算子。

用法: python3 tools/fix_lprnet_onnx.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    g = m.graph

    # 定位问题节点：ReduceMean(axes=[2])，输入 [1,68,4,18]
    target = None
    for n in g.node:
        if n.op_type == "ReduceMean":
            axes = [a for a in n.attribute if a.name == "axes"]
            if axes and list(axes[0].ints) == [2]:
                target = n
                break
    if target is None:
        raise SystemExit("target ReduceMean(axes=[2]) not found")

    in_name = target.input[0]          # '138'
    out_name = target.output[0]        # 'output'
    print(f"[fix] {target.name}: {in_name} -> {out_name}")

    H, W, C = 4, 18, 68                 # 已知固定 shape
    ap_out = "62_avgpool_out"
    shape_init = helper.make_tensor(
        "62_reshape_shape", TensorProto.INT64, [3], [1, C, W])

    ap = helper.make_node(
        "AveragePool", [in_name], [ap_out],
        name="ReplaceMean_AvgPool", kernel_shape=[H, 1],
        pads=[0, 0, 0, 0], strides=[H, 1])
    rs = helper.make_node(
        "Reshape", [ap_out, "62_reshape_shape"], [out_name],
        name="ReplaceMean_Reshape")

    # 原位替换：删旧节点，插两个新节点
    idx = list(g.node).index(target)
    g.node.remove(target)
    g.node.insert(idx, rs)
    g.node.insert(idx, ap)
    g.initializer.append(shape_init)

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"[ok] saved {dst}")


if __name__ == "__main__":
    main()
