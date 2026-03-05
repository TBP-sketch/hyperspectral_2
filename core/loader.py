from __future__ import annotations

"""
数据加载器（与 UI 解耦）。

目前提供：
- load_mat_file(file_path): 支持 MATLAB .mat（普通 v7 及 v7.3 HDF5）解析。

设计目标：
- 尽可能“智能”地从 .mat 中找出高光谱立方体（H, W, B）；
- 同时提取波长信息（wavelength / bands / lambda），找不到则生成默认序号波长；
- 解析过程对无关字段（字符串、结构体、cell 等）保持健壮，不轻易崩溃。
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


MAT_CUBE_KEYWORDS = (
    "data",
    "reflectance",
    "image",
    "cube",
    "indian_pines",
    "indianpines",
    "pavia",
    "pavia_university",
    "paviauniversity",
)

MAT_WAVELENGTH_KEYWORDS = ("wavelength", "wavelengths", "bands", "lambda", "wl")


def _is_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _score_key(name: str, keywords: Iterable[str]) -> int:
    n = name.lower()
    score = 0
    for k in keywords:
        if k in n:
            score += 10
    # 常见 MATLAB 私有键应降低优先级
    if n.startswith("__"):
        score -= 100
    return score


def _as_1d_float(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).reshape(-1)
    return a.astype(np.float64, copy=False)


def _try_extract_wavelengths(d: Dict[str, Any], bands: int) -> Optional[np.ndarray]:
    """
    从 .mat 字典中提取波长（1D 数组）。
    返回 None 表示未找到。
    """
    candidates: List[Tuple[int, str, np.ndarray]] = []
    for k, v in d.items():
        if not _is_numeric_ndarray(v):
            continue
        if np.asarray(v).ndim != 1:
            continue
        score = _score_key(k, MAT_WAVELENGTH_KEYWORDS)
        candidates.append((score, k, np.asarray(v)))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    for _score, _k, arr in candidates:
        wl = _as_1d_float(arr)
        if wl.size == bands:
            return wl

    # 有些数据集把 bands 写成 1..B
    for _score, _k, arr in candidates:
        wl = _as_1d_float(arr)
        if wl.size > 0:
            return wl

    return None


def _find_cube_candidates(d: Dict[str, Any]) -> List[Tuple[int, str, np.ndarray]]:
    """
    遍历字典，找出三维数值数组候选。
    返回列表：[(score, key, array)]
    """
    candidates: List[Tuple[int, str, np.ndarray]] = []
    for k, v in d.items():
        if not _is_numeric_ndarray(v):
            continue
        arr = np.asarray(v)
        if arr.ndim != 3:
            continue
        score = _score_key(k, MAT_CUBE_KEYWORDS)
        # 更倾向于 H*W 较大、B 适中的形状（典型高光谱）
        shape = arr.shape
        score += int(np.log10(max(1, shape[0] * shape[1])) * 2)
        candidates.append((score, k, arr))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates


def _normalize_cube(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    将候选三维数组归一到 (H, W, B)。

    支持输入形状：
    - (H, W, B) 直接返回
    - (B, H, W) 转置到 (H, W, B)
    其他形状：尽力推断（假设最小维度是 B），但若不可靠则按原样报错由上层处理。

    返回：(cube, layout_str)
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError("cube 必须是三维数组")

    s0, s1, s2 = a.shape

    # 常见：H,W,B
    if s2 <= max(s0, s1) and s0 > 1 and s1 > 1:
        return a, "(H,W,B)"

    # 常见：B,H,W
    if s0 <= max(s1, s2) and s1 > 1 and s2 > 1:
        return np.transpose(a, (1, 2, 0)), "(B,H,W)->(H,W,B)"

    # 兜底：把最小维当作 bands
    bands_axis = int(np.argmin([s0, s1, s2]))
    if bands_axis == 0:
        return np.transpose(a, (1, 2, 0)), "(?, ?, ?)->bands axis=0"
    if bands_axis == 1:
        return np.transpose(a, (0, 2, 1)), "(?, ?, ?)->bands axis=1"
    return a, "(?, ?, ?)->bands axis=2"


def _loadmat_v7(file_path: str) -> Dict[str, Any]:
    """使用 scipy.io.loadmat 读取普通 v7 .mat。"""
    try:
        from scipy import io as spio
    except Exception as e:
        raise ImportError("未安装 scipy，无法读取 .mat 文件。请先安装：pip install scipy") from e

    return spio.loadmat(file_path, struct_as_record=False, squeeze_me=True)


def _loadmat_v73_h5(file_path: str) -> Dict[str, Any]:
    """
    MATLAB v7.3（HDF5）回退读取。

    为了复用上面的候选提取逻辑，这里把 HDF5 中的 dataset 读成 dict：
    - key 使用 HDF5 路径（去掉开头 '/'），例如 'data' 或 'group1/cube'
    - value 为 numpy ndarray
    """
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "该 .mat 可能为 MATLAB v7.3（HDF5），需要 h5py 才能读取。请先安装：pip install h5py"
        ) from e

    out: Dict[str, Any] = {}
    with h5py.File(file_path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            try:
                import h5py  # local for type
                if isinstance(obj, h5py.Dataset):
                    out[name] = np.asarray(obj)
            except Exception:
                return

        f.visititems(visitor)
    return out


def load_mat_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    读取 MATLAB .mat 文件，并返回：
    - data_cube: (H, W, B) float32
    - wavelengths: (B,) float64（若无则为 1..B）
    - source_key: 使用的变量名（用于 UI 提示）

    异常：
    - 若找不到三维数组，抛 ValueError(\"未在 .mat 文件中找到有效的高光谱数据立方体\")
    """
    # 1) 优先尝试 v7（scipy.io.loadmat）
    try:
        d = _loadmat_v7(file_path)
    except NotImplementedError as e:
        # scipy 对 v7.3 往往提示使用 h5py
        d = _loadmat_v73_h5(file_path)
    except Exception as e:
        # 仍可能是 v7.3 或损坏文件，尝试 h5py 回退
        try:
            d = _loadmat_v73_h5(file_path)
        except Exception:
            raise

    # 2) 找候选立方体
    candidates = _find_cube_candidates(d)
    if not candidates:
        raise ValueError("未在 .mat 文件中找到有效的高光谱数据立方体")

    # 3) 选中最优候选（若多个，由 UI 层决定是否弹窗选择；此处只返回最优）
    score, key, arr = candidates[0]
    cube, _layout = _normalize_cube(arr)

    # 4) 波长提取
    bands = cube.shape[2]
    wl = _try_extract_wavelengths(d, bands=bands)
    if wl is None or wl.size != bands:
        wl = np.arange(1, bands + 1, dtype=np.float64)

    return cube.astype(np.float32, copy=False), wl.astype(np.float64, copy=False), key


def list_mat_cube_candidates(file_path: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    供 UI 使用：列出 .mat 文件内所有三维数组候选（变量名与形状）。\n
    注意：此函数会读取 .mat；对超大文件可能较慢。
    """
    try:
        d = _loadmat_v7(file_path)
    except Exception:
        d = _loadmat_v73_h5(file_path)

    return [(k, tuple(np.asarray(v).shape)) for _s, k, v in _find_cube_candidates(d)]

