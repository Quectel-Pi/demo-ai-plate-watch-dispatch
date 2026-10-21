#!/usr/bin/env python3
"""RKNN NPU 后端：纯 ctypes 直调 librknnrt.so（不依赖 rknnlite 的 cpython 版本绑定）。

适用于板上 Python 3.13 + librknnrt 2.x。输入 NHWC float32（自动量化进 NPU），
输出 want_float=1 取回 float32。
"""
import ctypes
import numpy as np

# ---- rknn_api.h 关键常量 ----
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_NHWC = 1

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256


class rknn_input_output_num(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RKNNRuntime:
    def __init__(self, model_path: str, lib_path: str = "librknnrt.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        self.lib.rknn_query.argtypes = [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        self.lib.rknn_inputs_set.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(rknn_input)]
        self.lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        self.lib.rknn_outputs_get.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32,
            ctypes.POINTER(rknn_output), ctypes.c_void_p]
        self.lib.rknn_outputs_release.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(rknn_output)]
        self.lib.rknn_destroy.argtypes = [ctypes.c_uint64]

        # init：直接传模型路径（size=0）
        self.ctx = ctypes.c_uint64(0)
        path_b = model_path.encode()
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx),
            ctypes.cast(ctypes.c_char_p(path_b), ctypes.c_void_p),
            0, 0, None)
        if ret != 0:
            raise RuntimeError(f"rknn_init failed: {ret}")

        # 查询输入输出数量与属性
        num = rknn_input_output_num()
        self._query(RKNN_QUERY_IN_OUT_NUM, num)
        self.n_input, self.n_output = num.n_input, num.n_output

        self.in_attrs = []
        for i in range(self.n_input):
            attr = rknn_tensor_attr(index=i)
            self._query(RKNN_QUERY_INPUT_ATTR, attr)
            self.in_attrs.append(attr)
        self.out_attrs = []
        for i in range(self.n_output):
            attr = rknn_tensor_attr(index=i)
            self._query(RKNN_QUERY_OUTPUT_ATTR, attr)
            self.out_attrs.append(attr)
        self._out_shape = list(self.out_attrs[0].dims)[:self.out_attrs[0].n_dims]

    def _query(self, cmd, buf):
        """buf: ctypes.Structure 实例"""
        size = ctypes.sizeof(type(buf))
        ret = self.lib.rknn_query(self.ctx, cmd, ctypes.byref(buf), size)
        if ret != 0:
            raise RuntimeError(f"rknn_query({cmd}) failed: {ret}")

    def run(self, input_nhwc_f32: np.ndarray) -> np.ndarray:
        """输入 NHWC float32（RGB），返回输出 float32 ndarray"""
        data = np.ascontiguousarray(input_nhwc_f32, dtype=np.float32)
        inp = rknn_input(
            index=0, buf=data.ctypes.data_as(ctypes.c_void_p),
            size=data.nbytes, pass_through=0,
            type=RKNN_TENSOR_FLOAT32, fmt=RKNN_TENSOR_NHWC)
        ret = self.lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(inp))
        if ret != 0:
            raise RuntimeError(f"rknn_inputs_set failed: {ret}")

        ret = self.lib.rknn_run(self.ctx, None)
        if ret != 0:
            raise RuntimeError(f"rknn_run failed: {ret}")

        n_elems = self.out_attrs[0].n_elems
        out = rknn_output(want_float=1, is_prealloc=0, index=0)
        ret = self.lib.rknn_outputs_get(self.ctx, 1, ctypes.byref(out), None)
        if ret != 0:
            raise RuntimeError(f"rknn_outputs_get failed: {ret}")
        arr = np.ctypeslib.as_array(
            ctypes.cast(out.buf, ctypes.POINTER(ctypes.c_float)),
            shape=(n_elems,)).copy()
        self.lib.rknn_outputs_release(self.ctx, 1, ctypes.byref(out))
        return arr.reshape(self._out_shape)

    def close(self):
        if self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = ctypes.c_uint64(0)
