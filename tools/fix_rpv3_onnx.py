#!/usr/bin/env python3
"""rpv3 ONNX 手术：清除 100 个「输入+initializer 二重身」权重输入，只留真输入 data。

背景：HyperLPR3 导出时把权重同时挂成图输入和 initializer，
onnxruntime 默认用 initializer 值，RKNN 转换器却会因多输入报错。
"""
import onnx

SRC = '/home/q/.hyperlpr3/20230229/onnx/rpv3_mdict_160_r3.onnx'
DST = '/home/q/project/Demo/AI-Plate-Watch/models/rpv3_fixed.onnx'

m = onnx.load(SRC)
inits = {i.name for i in m.graph.initializer}
keep_inputs = [i for i in m.graph.input if i.name not in inits]
removed = len(m.graph.input) - len(keep_inputs)
del m.graph.input[:]
m.graph.input.extend(keep_inputs)
onnx.save(m, DST)
print(f'移除 {removed} 个权重输入, 保留 {[i.name for i in keep_inputs]}')
print(f'输出: {[o.name for o in m.graph.output]}')

# 验证：onnxruntime 原图 vs 手术图，输出应一致
import numpy as np
import onnxruntime as ort
x = np.random.rand(1, 3, 48, 160).astype(np.float32)
o1 = ort.InferenceSession(SRC).run(None, {'data': x})[0]
o2 = ort.InferenceSession(DST).run(None, {'data': x})[0]
print(f'随机输入输出 diff: {np.abs(o1 - o2).max():.2e}')
