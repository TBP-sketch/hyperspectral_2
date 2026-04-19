"""
全局背景图工具：等比铺满并居中裁剪；供主窗口 paintEvent 使用。
资源路径：assets/hyspec_background.png
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPixmap


def default_background_image_path() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "assets", "hyspec_background.png")


def load_background_pixmap(path: Optional[str] = None) -> QPixmap:
    p = path or default_background_image_path()
    if p and os.path.isfile(p):
        return QPixmap(p)
    return QPixmap()


def paint_background_cover(painter: QPainter, pixmap: QPixmap, x: int, y: int, w: int, h: int) -> None:
    """在矩形区域内绘制「覆盖裁剪」后的背景图。"""
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    w, h = max(w, 1), max(h, 1)
    if pixmap.isNull():
        painter.fillRect(x, y, w, h, QColor(0x2B, 0x2D, 0x30))
        return
    scaled = pixmap.scaled(
        w,
        h,
        Qt.KeepAspectRatioByExpanding,
        Qt.SmoothTransformation,
    )
    sw, sh = scaled.width(), scaled.height()
    sx = max(0, (sw - w) // 2)
    sy = max(0, (sh - h) // 2)
    painter.drawPixmap(x, y, w, h, scaled, sx, sy, w, h)
