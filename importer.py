"""
数据格式转换为 ENVI 的核心函数占位实现。

本模块中的 convert_to_envi 由用户根据实际需求自行实现。
当前仅提供函数签名与参数说明，调用时会抛出 NotImplementedError。

建议的 format_type 取值：
- 'geotiff'：GeoTIFF 光栅数据；
- 'hdf'：HDF4 / HDF5 数据；
- 'binary'：通用二进制文件。

对于 'binary'，kwargs 中建议包含：
- samples: int  列数
- lines: int    行数
- bands: int    波段数
- data_type: int    数据类型编码（例如：1=uint8, 2=uint16, 3=int16, 4=float32）
- interleave: str   'bsq' / 'bil' / 'bip'
- header_offset: int    文件头偏移字节数
- byte_order: int       0=小端，1=大端

progress_callback / log_callback：
- progress_callback(percentage: int)  接收 0~100 的进度百分比；
- log_callback(message: str)         接收日志字符串。

约定：输出 ENVI 文件的头文件路径为 f"{output_stem}.hdr"，
      数据文件路径可根据需要设定（例如 f"{output_stem}.img"）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def convert_to_envi(
    input_path: str,
    output_stem: str,
    format_type: str,
    progress_callback: Optional[Callable[[int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> None:
    """
    将多种数据格式转换为 ENVI 标准格式（.hdr + 数据文件）。

    该函数由用户在项目中自行实现；当前仅抛出 NotImplementedError。
    """
    if log_callback is not None:
        log_callback(
            "convert_to_envi 尚未实现。请在 importer.py 中根据实际需求实现转换逻辑。"
        )
    raise NotImplementedError("convert_to_envi 未实现。")

