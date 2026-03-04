from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    # 核心转换函数，由用户在 importer.py 中实现
    from importer import convert_to_envi
except Exception:  # 若导入失败，后续会在运行时给出友好提示
    convert_to_envi = None  # type: ignore[assignment]


class _ConvertWorker(QObject):
    """
    在后台线程中执行实际的转换工作。

    通过 progress / log / finished 信号，将进度与结果传回对话框。
    """

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # (success, hdr_path 或 "")

    def __init__(
        self,
        input_path: str,
        output_stem: str,
        format_type: str,
        kwargs: Dict[str, Any],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._input_path = input_path
        self._output_stem = output_stem
        self._format_type = format_type
        self._kwargs = kwargs

    def _on_progress(self, value: int) -> None:
        self.progress.emit(int(value))

    def _on_log(self, message: str) -> None:
        self.log.emit(str(message))

    def run(self) -> None:
        """工作线程入口。"""
        if convert_to_envi is None:
            self.log.emit("错误：convert_to_envi 函数不可用，请检查 importer.py。")
            self.finished.emit(False, "")
            return

        try:
            self.log.emit("开始转换为 ENVI 格式...")
            # 用户自定义的转换函数：本模块只负责参数与回调传递
            convert_to_envi(
                self._input_path,
                self._output_stem,
                self._format_type,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
                **self._kwargs,
            )
            # 若用户未在 convert_to_envi 中显式汇报 100%，这里做一次兜底
            self.progress.emit(100)
            hdr_path = self._output_stem + ".hdr"
            self.log.emit(f"转换完成，生成头文件：{hdr_path}")
            self.finished.emit(True, hdr_path)
        except Exception as e:
            self.log.emit(f"转换过程中出现错误：{e}")
            self.finished.emit(False, "")


class ConvertDialog(QDialog):
    """
    数据格式转换为 ENVI 的可视化对话框。

    支持三种源格式：
    - GeoTIFF（geotiff）
    - HDF4/5（hdf）
    - 通用二进制（binary）

    转换成功后，若勾选“自动加载”，会通过 envi_ready 信号把生成的 .hdr 路径
    通知主窗口，由主窗口完成加载与预览。
    """

    envi_ready = pyqtSignal(str)  # hdr_path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("数据格式转换为 ENVI")
        self.resize(600, 500)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[_ConvertWorker] = None

        # 动态参数控件引用（随格式变化重新创建，避免 Qt 删除后悬挂）
        self.edit_hdf_dataset: Optional[QLineEdit] = None
        self.btn_hdf_list: Optional[QPushButton] = None
        self.spin_bin_samples: Optional[QSpinBox] = None
        self.spin_bin_lines: Optional[QSpinBox] = None
        self.spin_bin_bands: Optional[QSpinBox] = None
        self.combo_bin_dtype: Optional[QComboBox] = None
        self.combo_bin_interleave: Optional[QComboBox] = None
        self.spin_bin_offset: Optional[QSpinBox] = None
        self.combo_bin_byteorder: Optional[QComboBox] = None

        self._init_ui()

    # ---------- 界面构建 ----------
    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # 输入文件
        input_layout = QHBoxLayout()
        self.edit_input = QLineEdit()
        btn_browse_input = QPushButton("浏览...")
        btn_browse_input.clicked.connect(self.browse_file)
        input_layout.addWidget(QLabel("输入文件："))
        input_layout.addWidget(self.edit_input, 1)
        input_layout.addWidget(btn_browse_input)
        main_layout.addLayout(input_layout)

        # 输出文件主名
        output_layout = QHBoxLayout()
        self.edit_output = QLineEdit()
        btn_browse_output = QPushButton("浏览...")
        btn_browse_output.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("输出主名："))
        output_layout.addWidget(self.edit_output, 1)
        output_layout.addWidget(btn_browse_output)
        main_layout.addLayout(output_layout)

        # 源格式选择
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("源格式："))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["GeoTIFF", "HDF4/5", "通用二进制"])
        self.combo_format.currentIndexChanged.connect(self.on_format_changed)
        fmt_layout.addWidget(self.combo_format, 1)
        main_layout.addLayout(fmt_layout)

        # 动态参数区域
        self.group_params = QGroupBox("格式参数")
        self.params_form = QFormLayout(self.group_params)
        main_layout.addWidget(self.group_params)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 日志区域
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        main_layout.addWidget(QLabel("转换日志："))
        main_layout.addWidget(self.text_log, 1)

        # 选项 & 按钮
        options_layout = QHBoxLayout()
        self.check_auto_load = QCheckBox("转换后自动加载到主窗口")
        self.check_auto_load.setChecked(True)
        options_layout.addWidget(self.check_auto_load)
        options_layout.addStretch(1)
        main_layout.addLayout(options_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self.btn_start = QPushButton("开始转换")
        self.btn_start.clicked.connect(self.start_conversion)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(btn_close)
        main_layout.addLayout(btn_layout)

        # 初始化格式参数区域（根据当前源格式创建相应控件）
        self.on_format_changed()

    def _clear_params_form(self) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)

    # ---------- 槽函数 ----------
    def browse_file(self) -> None:
        """选择输入文件，并自动填充输出主名。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "所有文件 (*.*)")
        if not path:
            return
        self.edit_input.setText(path)

        # 自动填充输出主名（同目录同主名）
        base_name = os.path.splitext(os.path.basename(path))[0]
        dir_name = os.path.dirname(path)
        output_stem = os.path.join(dir_name, base_name)
        self.edit_output.setText(output_stem)

    def browse_output(self) -> None:
        """选择输出文件主名（用户选择 .hdr 或任意文件名，我们取其不含扩展名的部分）。"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出 ENVI 头文件（程序会自动生成 .hdr 和数据文件）",
            "",
            "ENVI 头文件 (*.hdr);;所有文件 (*.*)",
        )
        if not path:
            return
        stem, _ = os.path.splitext(path)
        self.edit_output.setText(stem)

    def on_format_changed(self) -> None:
        """当源格式切换时，动态更新参数输入区域。"""
        self._clear_params_form()
        fmt_text = self.combo_format.currentText()

        if fmt_text == "GeoTIFF":
            self.group_params.setTitle("格式参数 - GeoTIFF")
            self.params_form.addRow(QLabel("无需额外参数。"))
        elif fmt_text == "HDF4/5":
            self.group_params.setTitle("格式参数 - HDF4/5")
            container_widget = QWidget()
            container_layout = QHBoxLayout(container_widget)
            container_layout.setContentsMargins(0, 0, 0, 0)
            # 每次格式切换时重新创建控件，避免 Qt 布局删除后使用悬挂指针
            self.edit_hdf_dataset = QLineEdit()
            self.btn_hdf_list = QPushButton("读取数据集列表")
            self.btn_hdf_list.clicked.connect(self.on_list_hdf_datasets)
            container_layout.addWidget(self.edit_hdf_dataset, 1)
            container_layout.addWidget(self.btn_hdf_list)

            self.params_form.addRow("数据集路径：", container_widget)
        else:  # 通用二进制
            self.group_params.setTitle("格式参数 - 通用二进制")
            # 重新创建各控件，防止被布局删除后继续使用同一实例导致崩溃
            self.spin_bin_samples = QSpinBox()
            self.spin_bin_samples.setRange(1, 10_000_000)
            self.spin_bin_samples.setValue(100)

            self.spin_bin_lines = QSpinBox()
            self.spin_bin_lines.setRange(1, 10_000_000)
            self.spin_bin_lines.setValue(100)

            self.spin_bin_bands = QSpinBox()
            self.spin_bin_bands.setRange(1, 10_000)
            self.spin_bin_bands.setValue(10)

            self.combo_bin_dtype = QComboBox()
            # 与 importer.convert_to_envi 中的数据类型编码约定：
            # 1: uint8, 2: uint16, 3: int16, 4: float32
            self.combo_bin_dtype.addItems(["uint8", "uint16", "int16", "float32"])

            self.combo_bin_interleave = QComboBox()
            self.combo_bin_interleave.addItems(["BSQ", "BIL", "BIP"])

            self.spin_bin_offset = QSpinBox()
            self.spin_bin_offset.setRange(0, 1_000_000_000)
            self.spin_bin_offset.setValue(0)

            self.combo_bin_byteorder = QComboBox()
            self.combo_bin_byteorder.addItems(["小端 (little-endian)", "大端 (big-endian)"])
            self.combo_bin_byteorder.setCurrentIndex(0)

            self.params_form.addRow("samples（列数）：", self.spin_bin_samples)
            self.params_form.addRow("lines（行数）：", self.spin_bin_lines)
            self.params_form.addRow("bands（波段数）：", self.spin_bin_bands)
            self.params_form.addRow("数据类型：", self.combo_bin_dtype)
            self.params_form.addRow("交错格式：", self.combo_bin_interleave)
            self.params_form.addRow("头偏移字节数：", self.spin_bin_offset)
            self.params_form.addRow("字节序：", self.combo_bin_byteorder)

    def on_list_hdf_datasets(self) -> None:
        """
        读取 HDF 文件中的数据集列表（简化版）。
        当前实现仅给出提示，实际遍历逻辑可在后续基于 h5py / pyhdf 等库扩展。
        """
        QMessageBox.information(
            self,
            "提示",
            "数据集列表功能暂未实现，请手动输入 HDF 内部数据集路径（例如：/Reflectance）。",
        )

    def _collect_params(self) -> Optional[Dict[str, Any]]:
        """从界面收集并校验参数，若有问题则弹出提示并返回 None。"""
        input_path = self.edit_input.text().strip()
        output_stem = self.edit_output.text().strip()

        if not input_path:
            QMessageBox.warning(self, "参数错误", "请输入输入文件路径。")
            return None
        if not os.path.isfile(input_path):
            QMessageBox.warning(self, "参数错误", "输入文件不存在。")
            return None
        if not output_stem:
            QMessageBox.warning(self, "参数错误", "请输入输出文件主名。")
            return None

        fmt_text = self.combo_format.currentText()
        if fmt_text == "GeoTIFF":
            format_type = "geotiff"
            kwargs: Dict[str, Any] = {}
        elif fmt_text == "HDF4/5":
            format_type = "hdf"
            dataset_path = self.edit_hdf_dataset.text().strip()
            if not dataset_path:
                QMessageBox.warning(self, "参数错误", "请输入 HDF 内部数据集路径。")
                return None
            kwargs = {"dataset_path": dataset_path}
        else:
            format_type = "binary"
            samples = int(self.spin_bin_samples.value())
            lines = int(self.spin_bin_lines.value())
            bands = int(self.spin_bin_bands.value())

            dtype_text = self.combo_bin_dtype.currentText()
            if dtype_text == "uint8":
                data_type = 1
            elif dtype_text == "uint16":
                data_type = 2
            elif dtype_text == "int16":
                data_type = 3
            else:  # float32
                data_type = 4

            interleave = self.combo_bin_interleave.currentText().lower()
            header_offset = int(self.spin_bin_offset.value())
            byte_order = 0 if self.combo_bin_byteorder.currentIndex() == 0 else 1

            kwargs = {
                "samples": samples,
                "lines": lines,
                "bands": bands,
                "data_type": data_type,
                "interleave": interleave,
                "header_offset": header_offset,
                "byte_order": byte_order,
            }

        return {
            "input_path": input_path,
            "output_stem": output_stem,
            "format_type": format_type,
            "kwargs": kwargs,
        }

    def start_conversion(self) -> None:
        """启动后台线程执行转换。"""
        params = self._collect_params()
        if params is None:
            return

        if convert_to_envi is None:
            QMessageBox.critical(
                self,
                "功能不可用",
                "importer.convert_to_envi 尚未实现或导入失败。\n"
                "请检查 importer.py 并实现相应的转换逻辑。",
            )
            return

        if self.worker_thread is not None:
            QMessageBox.warning(self, "正在转换", "当前已有转换任务在进行中。")
            return

        self.text_log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_start.setEnabled(False)

        self.worker_thread = QThread(self)
        self.worker = _ConvertWorker(
            params["input_path"],
            params["output_stem"],
            params["format_type"],
            params["kwargs"],
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.log.connect(self.on_worker_log)
        self.worker.finished.connect(self.on_worker_finished)

        # 线程结束后清理资源
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_worker_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def on_worker_log(self, message: str) -> None:
        self.text_log.append(str(message))

    def on_worker_finished(self, success: bool, hdr_path: str) -> None:
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.worker = None
        self.worker_thread = None

        if success:
            QMessageBox.information(self, "转换完成", "数据已成功转换为 ENVI 格式。")
            if self.check_auto_load.isChecked() and hdr_path:
                self.envi_ready.emit(hdr_path)
        else:
            QMessageBox.warning(self, "转换失败", "转换过程中发生错误，请查看日志。")

