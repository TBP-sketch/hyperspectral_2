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

样式通过外部 style.qss 加载（主色 #4CAF50，浅灰背景，圆角按钮等）。
"""

import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
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
    QToolBar,
    QWidget,
)

from pages import DataManager, DataLoadPage, ExportPage, PreprocessPage, VisualizationPage


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("高光谱数据处理平台")
        self.resize(1200, 720)

        # 数据管理器：在各页面之间共享
        self.data_manager = DataManager(self)

        # 构建 UI
        self._init_central_widgets()
        self._init_menus_and_toolbar()
        self._init_statusbar()

    # ---------- 中心区域：左侧导航 + 右侧工作区 ----------
    def _init_central_widgets(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        from PyQt5.QtWidgets import QHBoxLayout

        main_layout = QHBoxLayout(central)

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

        self.page_data_load = DataLoadPage(self.data_manager, self)
        self.page_preprocess = PreprocessPage(self.data_manager, self)
        self.page_export = ExportPage(self)
        self.page_visualization = VisualizationPage(self.data_manager, self)

        self.stack.addWidget(self.page_data_load)       # index 0
        self.stack.addWidget(self.page_preprocess)      # index 1
        self.stack.addWidget(self.page_export)          # index 2
        self.stack.addWidget(self.page_visualization)   # index 3

        main_layout.addWidget(self.nav_list)
        main_layout.addWidget(self.stack, 1)

        self.nav_list.setCurrentRow(0)

    def _add_nav_item(self, text: str, object_name: str) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, object_name)
        self.nav_list.addItem(item)

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
        open_action.triggered.connect(self.action_open_triggered)
        file_menu.addAction(open_action)

        save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "保存(&S)", self)
        save_action.setEnabled(False)  # 目前暂无保存功能
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("退出(&Q)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单（占位）
        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction("撤销", lambda: None).setEnabled(False)
        edit_menu.addAction("重做", lambda: None).setEnabled(False)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
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
        顶部“打开”按钮：实质上调用 DataLoadPage 的浏览+加载逻辑，
        并自动切换到“数据读取”模块。
        """
        self.nav_list.setCurrentRow(0)
        self.page_data_load.browse_file()

    def show_about_dialog(self) -> None:
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "关于",
            "高光谱数据处理平台（原型版）\n\n"
            "模块化架构：数据读取 / 预处理 / 导出 / 可视化\n"
            "基于 PyQt5 + Matplotlib 实现。",
        )


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

    # 加载外部样式表
    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
    load_qss(app, qss_path)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

