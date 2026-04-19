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
TECH_STACK = "Python, PyQt5, Matplotlib, NumPy, SciPy, Spectral, h5py, rasterio"


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
<h2>高光谱数据处理平台 — 使用说明</h2>
<p style="color:#555;">主窗口含背景图与四模块导航；菜单「视图」可切换<b>浅色 / 深色</b>主题。</p>

<h3>1. 数据读取</h3>
<p>在左侧选择 <b>📥 数据读取</b>，点击「浏览…」或菜单/工具栏<b>打开</b>，选择文件后点「加载」。读取过程中会显示「加载数据」提示框。</p>
<p><b>当前支持的格式（与文件对话框筛选一致）：</b></p>
<ul>
  <li><b>.npy</b> — NumPy 数组，形状须为三维 <b>(H, W, B)</b>（高 × 宽 × 波段）。</li>
  <li><b>.hdr</b> — ENVI 头文件；程序按头文件信息匹配同主名的 <b>.raw / .img / .dat</b> 等数据文件并读取。</li>
  <li><b>.mat</b> — MATLAB 数据；依赖 SciPy（及 v7.3 时可能需要 h5py）。若文件中存在多个三维数组，将提示选择其一；若解析到波长信息会保存供导出与可视化使用。</li>
</ul>
<p>加载成功后，「状态」区域显示数据形状 <b>(H, W, B)</b>；失败时会弹出错误说明。</p>

<h3>2. 预处理</h3>
<p>在 <b>⚙️ 预处理</b> 中：</p>
<ul>
  <li><b>辐射定标</b> — 可勾选「启用自动转换」按立方体最大值归一化到 [0,1]；或关闭后手动输入 <b>增益 (Gain)</b> 与 <b>偏移 (Offset)</b>，公式为 <b>R = Gain × DN + Offset</b>。点击「应用定标」后写回全局数据。</li>
  <li><b>简化大气校正</b> — 在「大气模式」「气溶胶类型」下拉框中选择选项后，点击「执行大气校正」。计算在后台线程执行，进度条与日志会更新；完成后数据同步到可视化与导出。</li>
</ul>
<p>模块下方为<b>日志</b>，可查看处理过程提示。</p>

<h3>3. 导出</h3>
<p>在 <b>💾 导出</b> 中：</p>
<ul>
  <li>勾选导出格式：<b>ENVI</b>（.hdr + .dat）、<b>CSV</b>、<b>GeoTIFF</b>（需本机已安装 <b>rasterio</b> 等依赖）。可同时勾选多种格式。</li>
  <li><b>导出范围</b>：「全图数据」将立方体展平为像素 × 波段导出；「当前选中像素的光谱曲线」依赖在可视化中已点击选取的像素。</li>
  <li>在「输出路径」中填写或浏览主文件名（无扩展名），点击「开始导出」。导出过程中会显示忙碌提示，日志区显示进度与结果路径。</li>
</ul>

<h3>4. 可视化</h3>
<p>在 <b>📊 可视化</b> 中：</p>
<ul>
  <li><b>波段选择</b> — 单波段模式下拉选择波段，左侧显示灰度图；可用「下限/上限百分位」滑块做对比度拉伸。</li>
  <li><b>假彩色合成</b> — 在 R/G/B 下拉框中分别选择波段，点击「生成假彩色图」。</li>
  <li><b>光谱曲线</b> — 在左侧图像上<b>左键单击</b>选取像素，右侧显示该像素光谱；可使用自定义按钮进行放大、缩小、重置、清除曲线、保存图像等（含 Matplotlib 工具栏中的缩放/平移）。</li>
</ul>
<p>无数据时画布为占位提示；加载数据后显示影像与曲线。</p>

<h3>菜单与快捷键</h3>
<ul>
  <li><b>文件 → 打开</b>（<b>Ctrl+O</b>）— 切换到数据读取并打开文件选择对话框。</li>
  <li><b>文件 → 保存</b>（<b>Ctrl+S</b>）— 切换到导出模块，请在导出页选择格式与路径后点击「开始导出」。</li>
  <li><b>文件 → 退出</b>（<b>Ctrl+Q</b>）— 退出程序。</li>
  <li><b>视图 → 深色模式</b> — 勾选为深色主题，取消为浅色主题。</li>
  <li><b>帮助 → 使用帮助</b>（<b>F1</b>）— 打开本说明；<b>关于</b> 查看版本与依赖说明。</li>
</ul>

<p style="color:#666; margin-top: 24px;">祝使用愉快！</p>
</body>
</html>
"""


class HelpWindow(QDialog):
    """
    帮助窗口：使用 QTextBrowser 显示 HTML 格式的使用说明。
    使用 QDialog 并带关闭按钮，确保在 Windows 下可正常关闭。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpWindow")
        self.setWindowTitle("使用帮助")
        self.setModal(False)
        self.resize(560, 520)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowCloseButtonHint
            | Qt.WindowSystemMenuHint
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(HELP_HTML)
        layout.addWidget(self.browser, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.setObjectName("secondaryButton")
        btn_close.clicked.connect(self.accept)
        row.addWidget(btn_close)
        layout.addLayout(row)
