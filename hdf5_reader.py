"""
HDF5 高光谱数据读取模块。

功能概述：
- 基础：打开 HDF5 文件（.h5 / .hdf5），递归列出所有 Groups / Datasets 结构；
- 核心：读取高光谱图像数据立方体（通常为 3 维：行 x 列 x 波段），
  并处理常见的 scale_factor / add_offset 属性，恢复为浮点物理量；
- 元数据：
  - 读取波段中心波长（典型路径为 '/wavelength'，也会在全文件中搜索名为 'wavelength' 的一维数据集）；
  - 尝试提取地理空间参考信息（EPSG、投影字符串、图像四角坐标）。

注意：不同机构生成的 HDF5 结构差异较大，本模块实现的是“尽可能通用且健壮”的一套启发式规则，
      对于特殊结构，可以在此基础上扩展对应的路径与字段解析。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except Exception as e:  # pragma: no cover - 运行期缺少依赖时给出明确错误
    h5py = None  # type: ignore[assignment]


class HDF5Error(Exception):
    """HDF5 读取相关通用异常。"""


@dataclass
class HDF5GeoInfo:
    """地理空间参考信息（尽量通用的结构）。"""

    epsg: Optional[int] = None
    projection: Optional[str] = None  # 可为 WKT / Proj4 / 其他字符串
    corners: Optional[List[Tuple[float, float]]] = None  # [(lon, lat), ...] 或 (x, y)


def _ensure_h5py_available() -> None:
    if h5py is None:
        raise ImportError(
            "未安装 h5py，无法读取 HDF5 文件。"
            "请在当前环境中安装：pip install h5py"
        )


def _dataset_tree_str(h5: "h5py.File") -> str:
    """
    递归列出文件中的 Groups 与 Datasets 结构，返回为多行字符串。
    类似 h5py.File.visititems 提供的遍历效果，便于调试与展示。
    """
    lines: List[str] = []

    def visitor(name: str, obj: "h5py.Dataset | h5py.Group") -> None:
        indent_level = name.count("/")
        indent = "  " * indent_level
        if isinstance(obj, h5py.Group):
            lines.append(f"{indent}[G] {name or '/'}")
        else:
            shape = obj.shape
            dtype = obj.dtype
            lines.append(f"{indent}[D] {name}  shape={shape}  dtype={dtype}")

    h5.visititems(visitor)
    return "\n".join(lines)


def _read_attr_number(attrs: "h5py.AttributeManager", key_candidates: List[str]) -> Optional[float]:
    """
    从 attributes 中读取一个数值型属性（第一命中者）。
    返回 float 或 None。
    """
    for k in key_candidates:
        if k in attrs:
            v = attrs[k]
            # 可能是标量，也可能是 0 维/1 元数组
            if isinstance(v, np.ndarray):
                if v.size == 0:
                    continue
                v = v.reshape(-1)[0]
            try:
                return float(v)
            except Exception:
                continue
    return None


def _find_dataset_by_name(h5: "h5py.File", target_name: str) -> Optional["h5py.Dataset"]:
    """
    在整个 HDF5 文件中按名称（不含路径）搜索数据集。
    返回第一个匹配的 Dataset，若未找到则返回 None。
    """
    found: Optional["h5py.Dataset"] = None

    def visitor(name: str, obj: "h5py.Dataset | h5py.Group") -> None:
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset) and os.path.basename(name) == target_name:
            found = obj

    h5.visititems(visitor)
    return found


def _find_reflectance_dataset(h5: "h5py.File") -> "h5py.Dataset":
    """
    尝试在 HDF5 中找到高光谱反射率数据立方体：
    - 优先使用常见路径 '/Reflectance'；
    - 否则在全文件中搜索名为 'Reflectance' 的三维数据集；
    - 若仍找不到，则抛出异常。
    """
    # 1. 直接尝试标准路径
    if "/Reflectance" in h5:
        ds = h5["/Reflectance"]
        if isinstance(ds, h5py.Dataset):
            if ds.ndim == 3:
                return ds

    # 2. 在全文件中搜索名为 'Reflectance' 的 3D 数据集
    candidate = _find_dataset_by_name(h5, "Reflectance")
    if candidate is not None and candidate.ndim == 3:
        return candidate

    raise HDF5Error(
        "未找到高光谱反射率数据集（期望路径 '/Reflectance' 或名称为 'Reflectance' 的三维数据集）。"
    )


def _find_wavelength_dataset(h5: "h5py.File") -> Optional["h5py.Dataset"]:
    """
    尝试获取波段中心波长数据集：
    - 优先 '/wavelength'；
    - 否则在全文件中搜索名为 'wavelength' 的一维数据集。
    若未找到，返回 None。
    """
    if "/wavelength" in h5:
        ds = h5["/wavelength"]
        if isinstance(ds, h5py.Dataset):
            return ds

    candidate = _find_dataset_by_name(h5, "wavelength")
    if candidate is not None and candidate.ndim == 1:
        return candidate

    return None


def _extract_geo_info(h5: "h5py.File") -> HDF5GeoInfo:
    """
    尝试从 HDF5 中提取地理空间参考信息。
    由于不同数据产品结构差异较大，这里采用启发式搜索：

    - EPSG:
      - root attrs: 'epsg', 'epsg_code', 'EPSG', 等；
      - 若为字符串或标量数值均转换为 int。
    - 投影字符串:
      - root attrs 中常见的 'proj4', 'projection', 'spatial_ref', 'Projection' 等；
    - 四角坐标:
      - 搜索名称中包含 'corner' 的 2D 数据集：
        典型形状为 (4, 2) 或 (2, 4)；也接受 (N, 2)。
    """
    root_attrs = h5.attrs

    # EPSG
    epsg = None
    for key in ("epsg", "epsg_code", "EPSG", "epsgCode"):
        if key in root_attrs:
            v = root_attrs[key]
            if isinstance(v, (bytes, str)):
                v = v.decode() if isinstance(v, bytes) else v
                try:
                    epsg = int(v.strip())
                    break
                except Exception:
                    continue
            else:
                try:
                    epsg = int(v)
                    break
                except Exception:
                    continue

    # 投影字符串
    projection = None
    for key in ("proj4", "projection", "spatial_ref", "SpatialRef", "projection_definition"):
        if key in root_attrs:
            v = root_attrs[key]
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            projection = str(v)
            break

    # 四角坐标（尽量通用）
    corners: Optional[List[Tuple[float, float]]] = None

    def _try_parse_corner_ds(ds: "h5py.Dataset") -> Optional[List[Tuple[float, float]]]:
        arr = np.asarray(ds)
        if arr.ndim != 2:
            return None
        # (4, 2) 或 (N, 2)
        if arr.shape[1] == 2:
            return [(float(x), float(y)) for x, y in arr]
        # (2, 4) -> 转置后 (4, 2)
        if arr.shape[0] == 2 and arr.shape[1] >= 2:
            arr2 = arr.T
            return [(float(x), float(y)) for x, y in arr2]
        return None

    if corners is None:
        for name, obj in h5.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            if "corner" in name.lower():
                parsed = _try_parse_corner_ds(obj)
                if parsed:
                    corners = parsed
                    break

    return HDF5GeoInfo(epsg=epsg, projection=projection, corners=corners)


def read_hdf5_hypercube(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    读取 HDF5 高光谱数据文件。

    参数：
        path: HDF5 文件路径（.h5 / .hdf5）

    返回：
        (cube, info)
        - cube: np.ndarray，形状为 (H, W, B) = (行, 列, 波段)，用于直接显示；
        - info: dict，包含：
            - 'tree': 所有 Groups / Datasets 结构的字符串描述；
            - 'wavelengths': 波段中心波长（np.ndarray 或 None）；
            - 'geo': HDF5GeoInfo 实例（字典化后存储）；
            - 'raw_shape': 反射率数据集原始形状；
            - 'dataset_path': 实际使用的数据集路径；
            - 'scale_factor' / 'add_offset': 实际使用的缩放因子与偏移量。
    """
    _ensure_h5py_available()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 HDF5 文件：{path}")

    with h5py.File(path, "r") as h5:
        tree = _dataset_tree_str(h5)

        # 1. 找到反射率立方体
        ds_reflectance = _find_reflectance_dataset(h5)
        raw_shape = tuple(ds_reflectance.shape)

        # 2. 读取并应用缩放因子与偏移量
        data = ds_reflectance[...]

        scale = _read_attr_number(
            ds_reflectance.attrs, ["scale_factor", "Scale_Factor", "scaleFactor"]
        )
        offset = _read_attr_number(
            ds_reflectance.attrs, ["add_offset", "Add_Offset", "offset"]
        )

        data = data.astype(np.float32, copy=False)
        if scale is not None:
            data = data * float(scale)
        if offset is not None:
            data = data + float(offset)

        # 3. 将数据整理为 (行, 列, 波段) = (H, W, B)
        cube: np.ndarray
        if data.ndim != 3:
            raise HDF5Error(f"反射率数据集维度不是 3：shape={data.shape}")

        # 常见情况 1：行 x 列 x 波段，直接使用
        if data.shape[2] >= 10:  # 波段数通常远大于行/列，若第三维较大可认为是波段
            cube = data
        else:
            # 尝试使用波长长度来推断哪个维度是波段
            # 例如：bands x lines x samples 或 lines x bands x samples 等
            # 若无法可靠判断，默认最后一维为波段。
            cube = np.moveaxis(data, -1, -1)

        # 4. 波长信息
        ds_wl = _find_wavelength_dataset(h5)
        wavelengths = None
        if ds_wl is not None:
            wavelengths = np.asarray(ds_wl, dtype=np.float32)

        # 5. 地理空间信息
        geo = _extract_geo_info(h5)

        info: Dict[str, Any] = {
            "tree": tree,
            "wavelengths": wavelengths,
            "geo": {
                "epsg": geo.epsg,
                "projection": geo.projection,
                "corners": geo.corners,
            },
            "raw_shape": raw_shape,
            "dataset_path": ds_reflectance.name,
            "scale_factor": scale,
            "add_offset": offset,
        }

    return cube, info

