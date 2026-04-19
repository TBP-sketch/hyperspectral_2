# -*- coding: utf-8 -*-
"""
QMessageBox 封装。

Windows 默认 Qt 样式下消息框常由原生绘制，仅依赖全局 QSS 易出现纯黑内容区。
做法：1) 主程序使用 Fusion 样式；2) 在此对每个 QMessageBox 再套一层内联 QSS，保证灰蓝底生效。
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton, QWidget

# 与 style.qss / style_dark.qss 中 QMessageBox 段保持一致（内联强制生效）
_QSS_LIGHT = """
QMessageBox {
    background-color: #5A6B80;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 8px;
}
QMessageBox QLabel {
    background-color: #5A6B80;
    color: #F8FAFC;
    border: none;
}
QMessageBox QLabel#qt_msgbox_label {
    background-color: #5A6B80;
    color: #F8FAFC;
}
QMessageBox QLabel#qt_msgbox_icon_label {
    background-color: transparent;
}
QMessageBox QDialogButtonBox {
    background-color: #5A6B80;
    border: none;
}
QMessageBox QPushButton {
    background-color: #6D7F95;
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.65);
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 80px;
}
QMessageBox QPushButton:hover {
    background-color: #7D91A8;
    border: 1px solid rgba(255, 255, 255, 0.85);
}
QMessageBox QPushButton:pressed {
    background-color: #4A5568;
}
"""

_QSS_DARK = """
QMessageBox {
    background-color: #4A5568;
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 8px;
}
QMessageBox QLabel {
    background-color: #4A5568;
    color: #F8FAFC;
    border: none;
}
QMessageBox QLabel#qt_msgbox_label {
    background-color: #4A5568;
    color: #F8FAFC;
}
QMessageBox QLabel#qt_msgbox_icon_label {
    background-color: transparent;
}
QMessageBox QDialogButtonBox {
    background-color: #4A5568;
    border: none;
}
QMessageBox QPushButton {
    background-color: #5A6B80;
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 80px;
}
QMessageBox QPushButton:hover {
    background-color: #6D7F95;
    border: 1px solid rgba(255, 255, 255, 0.85);
}
QMessageBox QPushButton:pressed {
    background-color: #3E4A5C;
}
"""


def _is_dark_theme() -> bool:
    app = QApplication.instance()
    if app is None:
        return False
    v = app.property("hyperspectral_dark_theme")
    return bool(v) if v is not None else False


def _apply_msgbox_style_attributes(box: QMessageBox) -> None:
    box.setAttribute(Qt.WA_StyledBackground, True)
    for w in box.findChildren(QLabel):
        w.setAttribute(Qt.WA_StyledBackground, True)
    for w in box.findChildren(QPushButton):
        w.setAttribute(Qt.WA_StyledBackground, True)


def _prepare_message_box(mb: QMessageBox) -> None:
    """内联样式 + 属性，双保险。"""
    mb.setStyleSheet(_QSS_DARK if _is_dark_theme() else _QSS_LIGHT)
    _apply_msgbox_style_attributes(mb)


def dlg_warning(parent: Optional[QWidget], title: str, text: str) -> int:
    mb = QMessageBox(parent)
    mb.setWindowTitle(title)
    mb.setText(text)
    mb.setIcon(QMessageBox.Warning)
    mb.setStandardButtons(QMessageBox.Ok)
    mb.setDefaultButton(QMessageBox.Ok)
    _prepare_message_box(mb)
    return int(mb.exec_())


def dlg_information(parent: Optional[QWidget], title: str, text: str) -> int:
    mb = QMessageBox(parent)
    mb.setWindowTitle(title)
    mb.setText(text)
    mb.setIcon(QMessageBox.Information)
    mb.setStandardButtons(QMessageBox.Ok)
    mb.setDefaultButton(QMessageBox.Ok)
    _prepare_message_box(mb)
    return int(mb.exec_())


def dlg_critical(parent: Optional[QWidget], title: str, text: str) -> int:
    mb = QMessageBox(parent)
    mb.setWindowTitle(title)
    mb.setText(text)
    mb.setIcon(QMessageBox.Critical)
    mb.setStandardButtons(QMessageBox.Ok)
    mb.setDefaultButton(QMessageBox.Ok)
    _prepare_message_box(mb)
    return int(mb.exec_())
