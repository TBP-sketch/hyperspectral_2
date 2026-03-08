# -*- coding: utf-8 -*-
"""
ENVI 高光谱数据读取模块。

解析 ENVI 标准 .hdr 头文件，并读取对应的 .raw/.dat 二进制数据，
支持多种数据类型与字节序，返回形状为 (lines, samples, bands) 的 numpy 数组。
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 异常与常量
# ---------------------------------------------------------------------------

class EnviError(Exception):
    """ENVI 读取相关错误（文件缺失、解析失败等）。"""
    pass


# ENVI 数据类型编码 -> NumPy dtype（小端）
# 参考 ENVI 官方与 envi_writer 约定
_ENVI_DTYPE_MAP: dict[str | int, np.dtype] = {
    # 数字编码（字符串形式，头文件中常见）
    "1": np.dtype("uint8"),
    "2": np.dtype("int16"),
    "3": np.dtype("int32"),
    "4": np.dtype("float32"),
    "5": np.dtype("float64"),
    "6": np.dtype("complex64"),
    "9": np.dtype("complex128"),
    "12": np.dtype("uint16"),
    "13": np.dtype("uint32"),
    "14": np.dtype("int64"),
    "15": np.dtype("uint64"),
    # 名称形式（小写，兼容混合大小写）
    "byte": np.dtype("uint8"),
    "uint8": np.dtype("uint8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "long": np.dtype("int32"),
    "float32": np.dtype("float32"),
    "float": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "double": np.dtype("float64"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "int64": np.dtype("int64"),
    "uint64": np.dtype("uint64"),
}


# ---------------------------------------------------------------------------
# 1. 解析 .hdr 头文件
# ---------------------------------------------------------------------------

def parse_hdr(hdr_path: str) -> dict[str, Any]:
    """
    读取 .hdr 文件，解析键值对（如 samples, lines, bands, data type, byte order 等）。

    Args:
        hdr_path: .hdr 文件路径。

    Returns:
        包含头信息的字典，键为小写，值多为字符串（如 '512', 'float32', '0'）。

    Raises:
        EnviError: 文件不存在或解析失败。
    """
    if not os.path.isfile(hdr_path):
        raise EnviError(f"头文件不存在: {hdr_path!r}")

    with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # ENVI 头文件格式：KEY = VALUE 或 KEY = { ... }
    result: dict[str, Any] = {}
    key_pattern = re.compile(r"^(\w[\w\s]*?)\s*=\s*", re.MULTILINE | re.IGNORECASE)

    i = 0
    while i < len(text):
        match = key_pattern.match(text[i:])
        if not match:
            i += 1
            continue
        key_raw = match.group(1).strip()
        key = key_raw.lower().replace(" ", " ").strip()
        start = i + match.end()
        value_str, consumed = _parse_value(text, start)
        result[key] = value_str
        i = start + consumed

    return result


def _parse_value(text: str, start: int) -> tuple[Any, int]:
    """从 start 位置解析一个值（字符串、数字或 {} 块），返回 (值, 消耗字符数)。"""
    p = start
    while p < len(text) and text[p] in " \t\r\n":
        p += 1
    if p >= len(text):
        return "", p - start

    if text[p] == "{":
        # 多行/多值块
        depth = 1
        p += 1
        begin = p
        while p < len(text) and depth > 0:
            if text[p] == "{":
                depth += 1
            elif text[p] == "}":
                depth -= 1
            p += 1
        inner = text[begin : p - 1].strip()
        # 常见为逗号或换行分隔的列表，返回为单字符串（保留原始，便于 bands 等解析）
        return inner, p - start
    # 单行值
    end = p
    while end < len(text) and text[end] not in "\r\n":
        end += 1
    value = text[p:end].strip()
    return value, end - start


# ---------------------------------------------------------------------------
# 2. ENVI 数据类型字符串 -> NumPy dtype
# ---------------------------------------------------------------------------

def get_numpy_dtype(envi_dtype_str: str) -> np.dtype:
    """
    将 ENVI 数据类型字符串（如 '4', 'float32', 'uint16'）转换为 NumPy dtype。

    支持数字编码与名称形式，统一小写比较，兼容大小写混合输入。

    Args:
        envi_dtype_str: 头文件中的 data type 值（字符串）。

    Returns:
        对应的 np.dtype（无字节序标记，读取时再根据 byte order 处理）。

    Raises:
        EnviError: 未知的数据类型。
    """
    if not isinstance(envi_dtype_str, str):
        envi_dtype_str = str(envi_dtype_str).strip()
    raw = envi_dtype_str.strip()
    key = raw.lower()
    if key in _ENVI_DTYPE_MAP:
        return _ENVI_DTYPE_MAP[key].newbyteorder("=")
    raise EnviError(f"不支持的 ENVI 数据类型: {envi_dtype_str!r}")


# ---------------------------------------------------------------------------
# 3. 根据头信息读取 .raw 数据
# ---------------------------------------------------------------------------

def read_envi_data(raw_path: str, hdr_dict: dict[str, Any]) -> np.ndarray:
    """
    根据 hdr_dict 中的尺寸和数据类型，从 .raw 文件中读取数据，
    处理字节序，并按 interleave 重整为 (lines, samples, bands)。

    Args:
        raw_path: .raw 或 .dat 数据文件路径。
        hdr_dict: parse_hdr() 返回的头信息字典。

    Returns:
        形状为 (lines, samples, bands) 的 numpy 数组。

    Raises:
        EnviError: 文件不存在或尺寸/类型不匹配。
    """
    if not os.path.isfile(raw_path):
        raise EnviError(f"未找到对应的 .raw 文件，请确认文件名与 .hdr 一致: {raw_path!r}")

    samples = int(hdr_dict.get("samples", 0))
    lines = int(hdr_dict.get("lines", 0))
    bands = int(hdr_dict.get("bands", 0))
    if samples <= 0 or lines <= 0 or bands <= 0:
        raise EnviError(
            f"头文件中 samples/lines/bands 无效: samples={samples}, lines={lines}, bands={bands}"
        )

    dtype = get_numpy_dtype(str(hdr_dict.get("data type", "4")).strip())
    byte_order = str(hdr_dict.get("byte order", "0")).strip()
    interleave = str(hdr_dict.get("interleave", "bsq")).strip().lower() or "bsq"

    # 字节序：0 = 小端（Intel），1 = 大端（Network）
    sys_is_little = np.little_endian
    need_swap = (byte_order == "1" and sys_is_little) or (byte_order == "0" and not sys_is_little)

    with open(raw_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)

    expected_size = lines * samples * bands
    if data.size != expected_size:
        raise EnviError(
            f"数据大小与头文件不符: 文件 {data.size} 像素, 头文件 {expected_size} (L×S×B)"
        )

    if need_swap:
        data = data.byteswap()

    # 按存储格式 reshape，再统一为 (lines, samples, bands)
    if interleave == "bsq":
        # 存储顺序: [band0_all, band1_all, ...] -> (bands, lines, samples)
        data = data.reshape((bands, lines, samples))
        data = np.transpose(data, (1, 2, 0))  # -> (lines, samples, bands)
    elif interleave == "bil":
        # 存储顺序: [line0_band0..bandN, line1_...] -> (lines, bands, samples)
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (0, 2, 1))  # -> (lines, samples, bands)
    elif interleave == "bip":
        # 存储顺序: [pixel0_bands, pixel1_...] -> (lines, samples, bands)
        data = data.reshape((lines, samples, bands))
    else:
        raise EnviError(f"不支持的 interleave: {interleave!r}（期望 bsq / bil / bip）")

    return data


# ---------------------------------------------------------------------------
# 4. 主入口：从 .hdr 路径加载整组数据
# ---------------------------------------------------------------------------

def load_envi_dataset(hdr_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    给定 .hdr 文件路径，自动查找同名 .raw/.dat，解析头并读取数据。

    Args:
        hdr_path: .hdr 头文件路径。

    Returns:
        (data_array, header_dict)，数组形状为 (lines, samples, bands)。

    Raises:
        EnviError: 头文件不存在、.raw 不存在或解析/读取失败。
    """
    hdr_path = os.path.abspath(hdr_path)
    if not hdr_path.lower().endswith(".hdr"):
        raise EnviError(f"期望 .hdr 文件路径，得到: {hdr_path!r}")

    base = hdr_path[:-4]  # 去掉 .hdr
    raw_candidates = [base + ".raw", base + ".dat", base + ".RAW", base + ".DAT"]
    raw_path = None
    for p in raw_candidates:
        if os.path.isfile(p):
            raw_path = p
            break
    if raw_path is None:
        raise EnviError(
            "未找到对应的 .raw 文件，请确认文件名与 .hdr 一致。"
            f" 尝试路径: {raw_candidates[0]!r} 等。"
        )

    hdr_dict = parse_hdr(hdr_path)
    data = read_envi_data(raw_path, hdr_dict)
    return data, hdr_dict


# ---------------------------------------------------------------------------
# 5. 对外接口：支持传入 .hdr 或 .raw 路径
# ---------------------------------------------------------------------------

def read_envi(file_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    加载 ENVI 数据集，支持传入 .hdr 或 .raw/.dat 路径。

    - 若为 .hdr：直接按该头文件查找同名 .raw/.dat 并加载。
    - 若为 .raw/.dat：查找同名 .hdr 再加载。

    Returns:
        (data_array, header_dict)，数组形状为 (lines, samples, bands)。

    Raises:
        EnviError: 文件缺失或解析/读取失败。
    """
    file_path = os.path.abspath(file_path)
    low = file_path.lower()
    if low.endswith(".hdr"):
        return load_envi_dataset(file_path)
    if low.endswith(".raw") or low.endswith(".dat"):
        base = file_path[:-4]
        hdr_path = base + ".hdr"
        if not os.path.isfile(hdr_path):
            raise EnviError(
                "未找到对应的 .hdr 文件，请确认文件名与 .raw 一致。"
                f" 尝试路径: {hdr_path!r}"
            )
        return load_envi_dataset(hdr_path)
    raise EnviError(f"不支持的文件扩展名，请使用 .hdr 或 .raw/.dat: {file_path!r}")


# ---------------------------------------------------------------------------
# 示例用法
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 示例用法：可直接传入 .hdr 路径，或使用默认 test.hdr
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.hdr"
    data, hdr = load_envi_dataset(path)
    print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
