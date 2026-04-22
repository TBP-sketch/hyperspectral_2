from __future__ import annotations

"""
GeoTIFF 写出工具（依赖 rasterio）。

- write_geotiff_cube：将 (H, W, B) 高光谱立方体写为多波段 GeoTIFF；
- write_geotiff_spectrum：将长度为 B 的光谱向量写为 B 个波段、空间尺寸 1×1 的 GeoTIFF。

空间参考：当前工程内 DataManager 未携带真实投影信息，写出时使用像素坐标仿射变换
（原点左上、Y 轴向下与图像行一致），CRS 不写入；若日后接入地理配准，可在此扩展。
"""

from typing import Optional, Tuple

import os

import numpy as np

_RASTERIO_DTYPES = frozenset(
    ("uint8", "uint16", "int16", "int32", "uint32", "float32", "float64")
)


def _ensure_writable_dtype(arr: np.ndarray) -> np.ndarray:
    base = arr.dtype.name
    if base in _RASTERIO_DTYPES:
        return arr
    return arr.astype(np.float32, copy=False)


def _band_descriptions(bands: int, wavelengths_nm: Optional[np.ndarray]) -> Tuple[str, ...]:
    if wavelengths_nm is not None and len(wavelengths_nm) == bands:
        return tuple(f"Band {i + 1} ({float(wavelengths_nm[i]):.6g} nm)" for i in range(bands))
    return tuple(f"Band {i + 1}" for i in range(bands))


def write_geotiff_cube(
    output_stem: str,
    cube: np.ndarray,
    wavelengths_nm: Optional[np.ndarray] = None,
) -> str:
    """
    将 (lines, samples, bands) 即 (H, W, B) 写出为单文件多波段 GeoTIFF（.tif）。

    返回写出文件的完整路径。
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"GeoTIFF 运行时依赖加载失败（rasterio/GDAL）。原始错误：{e}"
        ) from e

    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"GeoTIFF 全图导出仅支持三维数组 (H,W,B)，当前维度为 {arr.ndim}")

    arr = _ensure_writable_dtype(arr)
    h, w, b = arr.shape

    out_path = output_stem + ".tif"
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 像素坐标：列向右、行向下；与 numpy 行主序一致
    transform = from_origin(0.0, float(h), 1.0, -1.0)
    data = np.transpose(arr, (2, 0, 1))

    wl = None
    if wavelengths_nm is not None and len(wavelengths_nm) == b:
        wl = np.asarray(wavelengths_nm, dtype=float)
    descriptions = _band_descriptions(b, wl)

    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": b,
        "dtype": arr.dtype,
        "transform": transform,
        "crs": None,
        "compress": "deflate",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i in range(b):
            dst.set_band_description(i + 1, descriptions[i])
        dst.update_tags(
            SOFTWARE="hyperspectral",
            DESCRIPTION="Hyperspectral cube export (H,W,B bands)",
        )

    return out_path


def write_geotiff_spectrum(
    output_stem: str,
    spectrum: np.ndarray,
    wavelengths_nm: Optional[np.ndarray] = None,
) -> str:
    """
    将长度 B 的光谱向量导出为 B 波段、1×1 像素的 GeoTIFF（每波段单像素）。

    便于在 GIS 中查看波段描述（波长）与像元值；空间位置为占位仿射。
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"GeoTIFF 运行时依赖加载失败（rasterio/GDAL）。原始错误：{e}"
        ) from e

    y = np.asarray(spectrum, dtype=float).reshape(-1)
    b = int(y.shape[0])
    if b == 0:
        raise ValueError("光谱数据为空。")

    arr = _ensure_writable_dtype(y)
    data = arr.reshape(b, 1, 1)

    out_path = output_stem + ".tif"
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    h, w = 1, 1
    transform = from_origin(0.0, float(h), 1.0, -1.0)

    wl = None
    if wavelengths_nm is not None and len(wavelengths_nm) == b:
        wl = np.asarray(wavelengths_nm, dtype=float)
    descriptions = _band_descriptions(b, wl)

    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": b,
        "dtype": arr.dtype,
        "transform": transform,
        "crs": None,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i in range(b):
            dst.set_band_description(i + 1, descriptions[i])
        dst.update_tags(
            SOFTWARE="hyperspectral",
            DESCRIPTION="Single-pixel spectrum export (B bands, 1x1)",
        )

    return out_path
