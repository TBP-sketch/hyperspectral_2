from __future__ import annotations

"""
主窗口模块：实现四模块架构的 MainWindow。

模块：
1. 数据读取 (Data Load)
2. 预处理   (Preprocessing)
3. 导出     (Export)
4. 可视化   (Visualization)

左侧使用 QListWidget 作为导航，右侧使用 QStackedWidget 作为主工作区。
顶部包含 QMenuBar / QToolBar，底部包含 QStatusBar。
支持全局快捷键与深色/浅色主题切换。

界面样式由 style.qss / style_dark.qss 提供；主要操作按钮需设置 objectName 为
「primaryButton」，浏览/关闭等次要按钮为「secondaryButton」（见各页面中的 QPushButton）。
"""

import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QStatusBar,
    QStackedWidget,
    QStyle,
    QStyleFactory,
    QToolBar,
    QWidget,
)

from app_background import load_background_pixmap, paint_background_cover
from pages import DataManager, DataLoadPage, ExportPage, PreprocessPage, VisualizationPage

try:
    from dialogs import AboutDialog, HelpWindow
except Exception:
    AboutDialog = None  # type: ignore
    HelpWindow = None  # type: ignore


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("高光谱数据处理平台")
        self.resize(1200, 720)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._background_pixmap: QPixmap = load_background_pixmap()

        # 主题：False=浅色，True=深色
        self._dark_theme = False
        self._qss_light = ""
        self._qss_dark = ""
        self._load_theme_styles()

        # 数据管理器：在各页面之间共享
        self.data_manager = DataManager(self)

        # 构建 UI
        self._init_central_widgets()
        self._init_menus_and_toolbar()
        self._init_statusbar()
        self._setup_shortcuts()

        # 帮助窗口单例（可复用）
        self._help_window: Optional[QWidget] = None

    # ---------- 中心区域：左侧导航 + 右侧工作区 ----------
    def _init_central_widgets(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWorkspace")
        central.setAttribute(Qt.WA_TranslucentBackground, True)
        central.setStyleSheet("#centralWorkspace { background: transparent; }")
        self.setCentralWidget(central)

        from PyQt5.QtWidgets import QHBoxLayout

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航栏
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setFixedWidth(180)
        self.nav_list.setSpacing(4)
        self.nav_list.setAlternatingRowColors(False)

        # 添加四个模块项
        self._add_nav_item("📥 数据读取", "data_load")
        self._add_nav_item("⚙️ 预处理", "preprocess")
        self._add_nav_item("💾 导出", "export")
        self._add_nav_item("📊 可视化", "visualization")

        self.nav_list.currentRowChanged.connect(self.on_nav_changed)

        # 右侧工作区：QStackedWidget
        self.stack = QStackedWidget()
        self.stack.setObjectName("mainStack")

        self.page_data_load = DataLoadPage(self.data_manager, self)
        self.page_preprocess = PreprocessPage(self.data_manager, self)
        # ExportPage 需要 DataManager，用于获取当前数据与光谱
        self.page_export = ExportPage(self.data_manager, self)
        self.page_visualization = VisualizationPage(self.data_manager, self)

        self.stack.addWidget(self.page_data_load)       # index 0
        self.stack.addWidget(self.page_preprocess)      # index 1
        self.stack.addWidget(self.page_export)          # index 2
        self.stack.addWidget(self.page_visualization)   # index 3

        main_layout.addWidget(self.nav_list)
        main_layout.addWidget(self.stack, 1)

        self.nav_list.setCurrentRow(0)

    def paintEvent(self, event) -> None:
        """整窗绘制背景图（菜单栏 / 工具栏 / 客户区均可见）。"""
        painter = QPainter(self)
        paint_background_cover(
            painter,
            self._background_pixmap,
            0,
            0,
            self.width(),
            self.height(),
        )
        painter.end()
        super().paintEvent(event)

    def _add_nav_item(self, text: str, object_name: str) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, object_name)
        self.nav_list.addItem(item)

    def _load_theme_styles(self) -> None:
        """加载浅色/深色两套 QSS，用于主题切换。"""
        base = os.path.dirname(os.path.abspath(__file__))
        for path, key in [
            (os.path.join(base, "style.qss"), "_qss_light"),
            (os.path.join(base, "style_dark.qss"), "_qss_dark"),
        ]:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        setattr(self, key, f.read())
                except Exception:
                    pass

    def _apply_theme(self, dark: bool) -> None:
        """应用主题（True=深色，False=浅色）。"""
        self._dark_theme = dark
        qss = self._qss_dark if dark else self._qss_light
        app = QApplication.instance()
        if app is not None:
            app.setProperty("hyperspectral_dark_theme", dark)
            if qss:
                app.setStyleSheet(qss)

    def _setup_shortcuts(self) -> None:
        """全局快捷键：Ctrl+O / Ctrl+S / Ctrl+Q / F1。"""
        from PyQt5.QtWidgets import QShortcut
        open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self.action_open_triggered)
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.action_save_triggered)
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.close)
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_help_window)

    def on_nav_changed(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        item = self.nav_list.item(index)
        if item is not None:
            self.statusBar().showMessage(f"当前模块：{item.text()}")

    # ---------- 菜单 & 工具栏 ----------
    def _init_menus_and_toolbar(self) -> None:
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        open_action = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "打开(&O)", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.action_open_triggered)
        file_menu.addAction(open_action)

        save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "保存(&S)", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.action_save_triggered)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单（占位）
        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction("撤销", lambda: None).setEnabled(False)
        edit_menu.addAction("重做", lambda: None).setEnabled(False)

        # 视图菜单 - 主题切换
        view_menu = menubar.addMenu("视图(&V)")
        self.theme_action = QAction("深色模式", self)
        self.theme_action.setCheckable(True)
        self.theme_action.setChecked(self._dark_theme)
        self.theme_action.triggered.connect(self._on_theme_toggled)
        view_menu.addAction(self.theme_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        help_action = QAction("使用帮助(&H)", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self.show_help_window)
        help_menu.addAction(help_action)
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # 工具栏
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("mainToolBar")
        # 使用统一的大图标尺寸
        icon_size = self.style().pixelMetric(QStyle.PM_LargeIconSize)
        toolbar.setIconSize(QSize(icon_size, icon_size))
        self.addToolBar(toolbar)

        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addSeparator()

        about_icon = self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        about_toolbar_action = QAction(about_icon, "关于", self)
        about_toolbar_action.triggered.connect(self.show_about_dialog)
        toolbar.addAction(about_toolbar_action)

    # ---------- 状态栏 ----------
    def _init_statusbar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("就绪")

    # ---------- 动作实现 ----------
    def action_open_triggered(self) -> None:
        """
        打开文件：切换到数据读取模块并打开文件对话框。
        """
        self.nav_list.setCurrentRow(0)
        self.page_data_load.browse_file()

    def action_save_triggered(self) -> None:
        """
        保存：切换到导出模块，便于用户执行导出操作。
        """
        self.nav_list.setCurrentRow(2)
        self.statusBar().showMessage("请在导出模块中选择格式与路径后点击「开始导出」。")

    def _on_theme_toggled(self, checked: bool) -> None:
        """深色/浅色主题切换。"""
        self._apply_theme(checked)

    def show_about_dialog(self) -> None:
        """显示关于对话框（含 Logo 占位、版本、作者、技术栈）。"""
        if AboutDialog is not None:
            dlg = AboutDialog(self)
            dlg.exec_()
        else:
            from ui_messages import dlg_information

            dlg_information(
                self,
                "关于",
                "高光谱数据处理平台 v1.0\n\n技术栈： Python, PyQt5, Matplotlib, Spectral",
            )

    def show_help_window(self) -> None:
        """显示帮助窗口（F1 或 帮助菜单）。"""
        if HelpWindow is None:
            return
        if self._help_window is None or not self._help_window.isVisible():
            self._help_window = HelpWindow(self)
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()


def load_qss(app: QApplication, qss_path: str) -> None:
    """从指定路径加载 QSS 样式文件。"""
    if not os.path.isfile(qss_path):
        return
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception:
        # 样式加载失败时静默忽略，不影响主程序运行
        pass


def main() -> None:
    app = QApplication(sys.argv)

    # Windows 默认样式下 QMessageBox 常走原生绘制，全局 QSS 易不生效；统一 Fusion 以便样式可控
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)

    app.setProperty("hyperspectral_dark_theme", False)

    # 加载外部样式表
    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
    load_qss(app, qss_path)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

