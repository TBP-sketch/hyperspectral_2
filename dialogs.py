# -*- coding: utf-8 -*-
"""
关于与帮助对话框，用于参赛展示与用户引导。
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# 软件信息常量，便于统一修改
APP_NAME = "高光谱数据处理平台"
APP_VERSION = "v1.0"
APP_AUTHOR = "参赛团队"
TECH_STACK = "Python, PyQt5, Matplotlib, Spectral, NumPy, h5py"


class AboutDialog(QDialog):
    """
    关于对话框：显示软件名称、版本、作者、技术栈及 Logo 占位。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setFixedSize(420, 360)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Logo 占位符（可用图片路径或占位文字）
        logo_label = QLabel()
        logo_label.setFixedSize(80, 80)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet(
            "background-color: #4CAF50; border-radius: 8px; color: white; font-size: 10pt;"
        )
        logo_label.setText("HSI\nLogo")
        logo_label.setScaledContents(False)
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        # 软件名称与版本
        name_label = QLabel(APP_NAME)
        name_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(name_label, alignment=Qt.AlignCenter)

        version_label = QLabel(APP_VERSION)
        version_label.setStyleSheet("color: #666;")
        layout.addWidget(version_label, alignment=Qt.AlignCenter)

        # 作者
        author_label = QLabel(f"作者：{APP_AUTHOR}")
        layout.addWidget(author_label, alignment=Qt.AlignCenter)

        # 技术栈
        tech_label = QLabel(f"技术栈：{TECH_STACK}")
        tech_label.setWordWrap(True)
        tech_label.setStyleSheet("color: #444; font-size: 10pt;")
        layout.addWidget(tech_label, alignment=Qt.AlignCenter)

        layout.addStretch(1)

        # 确定按钮
        btn = QPushButton("确定")
        btn.setObjectName("primaryButton")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# 帮助内容：简易 HTML（可由 Markdown 转 HTML 或直接写 HTML）
HELP_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Microsoft YaHei, SimHei, Arial; padding: 12px;">
<h2>高光谱数据处理平台 - 使用说明</h2>

<h3>1. 数据读取</h3>
<p>在左侧点击 <b>📥 数据读取</b>，通过「浏览」选择高光谱数据文件。支持格式：</p>
<ul>
  <li><b>.npy</b> — NumPy 三维数组 (H, W, B)</li>
  <li><b>.hdr</b> — ENVI 头文件（自动匹配同主名 .raw/.img/.dat）</li>
  <li><b>.h5 / .hdf5</b> — HDF5（自动读取 /Reflectance 及波长等元数据）</li>
  <li><b>.raw</b> — 通用二进制（需在弹窗中填写行/列/波段数、数据类型、交错方式等）</li>
</ul>
<p>加载成功后，可在「状态」区看到数据形状。</p>

<h3>2. 预处理</h3>
<p>在 <b>⚙️ 预处理</b> 模块可进行：</p>
<ul>
  <li><b>辐射定标</b> — 自动按最大值归一化或手动设置增益/偏移，将 DN 转为反射率。</li>
  <li><b>大气校正</b> — 选择大气模式与气溶胶类型后点击「执行大气校正」。</li>
</ul>
<p>处理结果会同步到可视化与导出模块。</p>

<h3>3. 导出</h3>
<p>在 <b>💾 导出</b> 中可选择：</p>
<ul>
  <li>导出为 <b>ENVI</b>（.hdr + .dat）、<b>CSV</b> 或 <b>GeoTIFF</b>（.tif）；</li>
  <li>范围可选「全图」或「当前选中像素的光谱曲线」；</li>
  <li>也可打开「数据格式转换为 ENVI」工具，将 GeoTIFF/ HDF/ 二进制转为 ENVI。</li>
</ul>

<h3>4. 可视化</h3>
<p>在 <b>📊 可视化</b> 中可：</p>
<ul>
  <li>选择单波段查看灰度图，并调节对比度拉伸（百分位）；</li>
  <li>使用 <b>假彩色合成</b> 选择 R/G/B 波段生成彩色图；</li>
  <li>在图像上 <b>左键点击</b> 查看该像素的光谱曲线；</li>
  <li>使用工具栏或滚轮缩放、中/右键拖拽平移、清除曲线、保存图像等。</li>
</ul>

<h3>快捷键</h3>
<ul>
  <li><b>Ctrl+O</b> — 打开文件</li>
  <li><b>Ctrl+S</b> — 保存（若已实现）</li>
  <li><b>Ctrl+Q</b> — 退出程序</li>
  <li><b>F1</b> — 打开本帮助</li>
</ul>

<p style="color:#666; margin-top: 24px;">祝使用愉快！</p>
</body>
</html>
"""


class HelpWindow(QWidget):
    """
    帮助窗口：使用 QTextBrowser 显示 HTML 格式的使用说明。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("使用帮助")
        self.resize(560, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(HELP_HTML)
        layout.addWidget(self.browser)
