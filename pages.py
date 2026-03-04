from __future__ import annotations

"""
页面模块：包含四个功能页：
- DataLoadPage       ：数据读取
- PreprocessPage     ：预处理（占位）
- ExportPage         ：导出（占位）
- VisualizationPage  ：可视化

以及简单的数据管理器 DataManager，用于在页面间共享高光谱数据。
"""

from typing import Any, Optional

import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Qt5Agg")

# 全局字体与负号设置（与原 app 保持一致）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False


# ---------- 依赖模块（ENVI / HDF5 / 转换对话框） ----------
try:
    from envi_reader import read_envi
except Exception:  # 若导入失败，相关功能会在运行时报友好错误
    read_envi = None  # type: ignore

try:
    from hdf5_reader import read_hdf5_hypercube
except Exception:
    read_hdf5_hypercube = None  # type: ignore

try:
    from convert_dialog import ConvertDialog
except Exception:
    ConvertDialog = None  # type: ignore


# ---------- 数据管理器 ----------
class DataManager(QObject):
    """
    简单的数据管理器：在页面间共享高光谱数据。

    当前只管理一个 numpy 数组 data（形状 (H, W, B)），后续可扩展为
    同时管理元数据（如波长、地理信息等）。
    """

    data_changed = pyqtSignal(np.ndarray)
    spectrum_changed = pyqtSignal(np.ndarray)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None
        self._current_spectrum: Optional[np.ndarray] = None
        self._wavelengths: Optional[np.ndarray] = None

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data

    def set_data(self, arr: np.ndarray) -> None:
        self._data = arr
        self.data_changed.emit(arr)

    @property
    def current_spectrum(self) -> Optional[np.ndarray]:
        return self._current_spectrum

    def set_current_spectrum(self, spectrum: np.ndarray) -> None:
        self._current_spectrum = spectrum
        self.spectrum_changed.emit(spectrum)

    @property
    def wavelengths(self) -> Optional[np.ndarray]:
        return self._wavelengths

    def set_wavelengths(self, wl: Optional[np.ndarray]) -> None:
        """
        设置波长数组（可选）。
        长度应与波段数一致，若不一致，上层在使用前需自行检查。
        """
        if wl is None:
            self._wavelengths = None
        else:
            self._wavelengths = np.asarray(wl, dtype=float)


# ---------- Matplotlib 画布 ----------
class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout()
        super().__init__(self.fig)
        self.setParent(parent)


# ---------- RAW 参数对话框 ----------
class RawFileDialog(QDialog):
    """RAW 文件参数输入对话框"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RAW 文件参数设置")
        self.setModal(True)

        layout = QGridLayout(self)

        # 高度
        layout.addWidget(QLabel("高度 (H):"), 0, 0)
        self.edit_height = QLineEdit("100")
        layout.addWidget(self.edit_height, 0, 1)

        # 宽度
        layout.addWidget(QLabel("宽度 (W):"), 1, 0)
        self.edit_width = QLineEdit("100")
        layout.addWidget(self.edit_width, 1, 1)

        # 波段数
        layout.addWidget(QLabel("波段数 (B):"), 2, 0)
        self.edit_bands = QLineEdit("10")
        layout.addWidget(self.edit_bands, 2, 1)

        # 数据类型
        layout.addWidget(QLabel("数据类型:"), 3, 0)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["uint8", "uint16", "int16", "float32", "float64"])
        self.dtype_combo.setCurrentText("float32")
        layout.addWidget(self.dtype_combo, 3, 1)

        # 字节顺序
        layout.addWidget(QLabel("字节顺序:"), 4, 0)
        self.byteorder_combo = QComboBox()
        self.byteorder_combo.addItems(["little-endian (<)", "big-endian (>)"])
        self.byteorder_combo.setCurrentText("little-endian (<)")
        layout.addWidget(self.byteorder_combo, 4, 1)

        # 数据排列方式
        layout.addWidget(QLabel("数据排列:"), 5, 0)
        self.interleave_combo = QComboBox()
        self.interleave_combo.addItems(
            [
                "BSQ (Band Sequential)",
                "BIL (Band Interleaved by Line)",
                "BIP (Band Interleaved by Pixel)",
            ]
        )
        self.interleave_combo.setCurrentText("BSQ (Band Sequential)")
        layout.addWidget(self.interleave_combo, 5, 1)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 6, 0, 1, 2)

        self.setLayout(layout)

    def get_params(self) -> dict:
        """获取用户输入的参数"""
        try:
            height = int(self.edit_height.text())
            width = int(self.edit_width.text())
            bands = int(self.edit_bands.text())
            dtype_str = self.dtype_combo.currentText()
            byteorder_str = (
                "<" if "little" in self.byteorder_combo.currentText() else ">"
            )
            interleave = self.interleave_combo.currentText().split()[0]  # BSQ/BIL/BIP

            # 构建完整的 dtype 字符串（包含字节顺序）
            dtype = np.dtype(byteorder_str + dtype_str)

            return {
                "height": height,
                "width": width,
                "bands": bands,
                "dtype": dtype,
                "interleave": interleave,
            }
        except ValueError as e:
            raise ValueError(f"参数输入错误：{e}")


# ---------- 页面：数据读取 ----------
class DataLoadPage(QWidget):
    """
    数据读取页面：负责选择并加载高光谱数据文件。

    - 支持 .npy / .raw / .hdr(ENVI) / .h5/.hdf5(HDF5)
    - 读取成功后，通过 DataManager 将 numpy 数组广播给其他页面
    """

    def __init__(self, data_manager: DataManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.data_manager = data_manager

        self.info_label = QLabel("未加载数据")
        self.info_label.setWordWrap(True)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 文件选择区域
        file_group = QGroupBox("数据文件")
        file_layout = QHBoxLayout(file_group)

        self.edit_path = QLineEdit()
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self.browse_file)
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(self.load_current_file)

        file_layout.addWidget(self.edit_path, 1)
        file_layout.addWidget(btn_browse)
        file_layout.addWidget(btn_load)

        layout.addWidget(file_group)

        # 可选：数据格式转换为 ENVI 的入口
        btn_convert = QPushButton("数据格式转换为 ENVI...")
        btn_convert.clicked.connect(self.open_convert_dialog)
        layout.addWidget(btn_convert)

        # 状态信息
        info_group = QGroupBox("状态")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)

        layout.addStretch(1)

    # ---- 文件选择 & 加载 ----
    def browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择高光谱数据文件",
            "",
            "NumPy 数组 (*.npy);;ENVI 头文件 (*.hdr);;HDF5 文件 (*.h5 *.hdf5);;RAW 文件 (*.raw);;所有文件 (*.*)",
        )
        if not path:
            return
        self.edit_path.setText(path)
        self.load_file(path)

    def load_current_file(self) -> None:
        path = self.edit_path.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择要加载的文件。")
            return
        self.load_file(path)

    def load_file(self, file_path: str) -> None:
        """核心加载逻辑（与原 HyperSpectralViewer.open_file 类似，但不涉及绘图）。"""
        if not file_path:
            return

        try:
            if file_path.lower().endswith(".npy"):
                arr = np.load(file_path)
                if arr.ndim == 3:
                    data = arr.astype(np.float32)
                elif arr.ndim == 2:
                    raise ValueError("期望数据形状为 (H, W, B)，当前为 2 维。")
                else:
                    raise ValueError(f"不支持的数据维度：{arr.shape}")

            elif file_path.lower().endswith(".hdr"):
                if read_envi is None:
                    raise ImportError(
                        "ENVI 读取模块不可用（envi_reader.py 导入失败）。请检查文件是否存在及依赖是否安装。"
                    )

                arr, header = read_envi(file_path)

                interleave = str(header.get("interleave", "")).strip().lower()
                if interleave == "bsq":
                    cube = np.transpose(arr, (1, 2, 0))
                elif interleave == "bil":
                    cube = np.transpose(arr, (0, 2, 1))
                elif interleave == "bip":
                    cube = arr
                else:
                    raise ValueError(
                        f"不支持的数据排列方式：{interleave!r}（期望 bsq/bil/bip）"
                    )

                data = cube.astype(np.float32)

            elif file_path.lower().endswith(".h5") or file_path.lower().endswith(
                ".hdf5"
            ):
                if read_hdf5_hypercube is None:
                    raise ImportError(
                        "HDF5 读取模块不可用（hdf5_reader.py 导入失败或未安装 h5py）。"
                    )
                cube, _info = read_hdf5_hypercube(file_path)
                data = cube.astype(np.float32)

            elif file_path.lower().endswith(".raw"):
                dialog = RawFileDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    params = dialog.get_params()
                    data = self._load_raw_file(file_path, params).astype(np.float32)
                else:
                    return

            else:
                raise ValueError(
                    "不支持的文件格式。支持格式：.npy, .raw, .hdr(ENVI), .h5/.hdf5(HDF5)"
                )

        except Exception as e:
            msg = str(e)
            if "不支持的文件格式" in msg:
                ui_msg = "当前格式暂不支持"
            elif "文件大小不匹配" in msg:
                ui_msg = "文件大小不匹配"
            elif "数据大小不匹配" in msg:
                ui_msg = "数据大小不匹配"
            else:
                ui_msg = f"加载失败：{msg}"

            QMessageBox.critical(self, "加载高光谱数据失败", ui_msg)
            self.info_label.setText("未加载数据")
            return

        # 广播数据
        self.data_manager.set_data(data)

        h, w, b = data.shape
        self.info_label.setText(f"数据已加载：形状 (H, W, B) = ({h}, {w}, {b})")

    def _load_raw_file(self, file_path: str, params: dict) -> np.ndarray:
        """读取 RAW 二进制文件，并返回 (H, W, B) 数组。"""
        height = params["height"]
        width = params["width"]
        bands = params["bands"]
        dtype = params["dtype"]
        interleave = params["interleave"]

        with open(file_path, "rb") as f:
            raw_data = np.fromfile(f, dtype=dtype)

        expected_size = height * width * bands
        if raw_data.size != expected_size:
            raise ValueError(
                f"文件大小不匹配！\n"
                f"文件包含 {raw_data.size} 个元素，\n"
                f"但根据参数 (H={height}, W={width}, B={bands}) 期望 {expected_size} 个元素。"
            )

        if interleave == "BSQ":
            raw_data = raw_data.reshape((bands, height, width))
            arr = np.transpose(raw_data, (1, 2, 0))
        elif interleave == "BIL":
            raw_data = raw_data.reshape((height, bands, width))
            arr = np.transpose(raw_data, (0, 2, 1))
        elif interleave == "BIP":
            arr = raw_data.reshape((height, width, bands))
        else:
            raise ValueError(f"不支持的数据排列方式：{interleave}")

        return arr

    def open_convert_dialog(self) -> None:
        """打开数据格式转换为 ENVI 的对话框。"""
        if ConvertDialog is None:
            QMessageBox.critical(
                self,
                "功能不可用",
                "ConvertDialog 未能导入，请检查 convert_dialog.py 是否存在且无语法错误。",
            )
            return

        dlg = ConvertDialog(self)

        def _on_envi_ready(hdr_path: str) -> None:
            # 转换完成后自动加载 ENVI 文件
            if not hdr_path:
                return
            self.edit_path.setText(hdr_path)
            self.load_file(hdr_path)

        dlg.envi_ready.connect(_on_envi_ready)
        dlg.exec_()


class _AtmosWorker(QObject):
    """
    大气校正工作线程中的实际执行者。

    为了避免阻塞 UI，把计算放到 QThread 中，由该对象负责具体处理。
    这里为了示例，仅实现一个“模拟的大气校正”流程：对数据立方体应用简单的缩放。
    如果项目中安装并正确配置了 Py6S，可以在此处接入真实计算逻辑。
    """

    progress = pyqtSignal(int)          # 进度：0-100
    log = pyqtSignal(str)              # 日志信息
    finished = pyqtSignal(bool, object)  # (是否成功, 结果数组或 None)

    def __init__(
        self,
        cube: np.ndarray,
        atm_profile: str,
        aero_profile: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._cube = cube
        self._atm_profile = atm_profile
        self._aero_profile = aero_profile

    def run(self) -> None:
        """
        工作入口函数：在 QThread 中调用。
        """
        try:
            self.log.emit(f"开始大气校正：大气模式={self._atm_profile}，气溶胶类型={self._aero_profile}")

            # 这里不强依赖 Py6S，仅做一个简化示例：
            # - 假设大气校正相当于对每个波段乘以一个系数（0.9 ~ 1.1 之间），
            #   以模拟“校正后亮度轻微变化”的效果。
            cube = self._cube.astype(np.float32, copy=True)
            h, w, b = cube.shape

            for i in range(b):
                # 简单的伪系数：根据索引平滑变化
                factor = 0.9 + 0.2 * (i / max(b - 1, 1))
                cube[:, :, i] *= factor

                # 更新进度
                progress = int((i + 1) / b * 100)
                self.progress.emit(progress)

            self.log.emit("大气校正完成。")
            self.finished.emit(True, cube)
        except Exception as e:
            self.log.emit(f"大气校正失败：{e}")
            self.finished.emit(False, None)


# ---------- 页面：预处理 ----------
class PreprocessPage(QWidget):
    """
    预处理模块：
    1. 辐射定标：DN -> 反射率（线性变换：Reflectance = Gain * DN + Offset）
    2. 简化大气校正：暴露大气模式 / 气溶胶类型，其余参数使用默认值或在工作线程中模拟实现。
    """

    def __init__(self, data_manager: DataManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data_manager = data_manager

        # 当前数据立方体引用（形状 (H, W, B)），由 DataManager 提供
        self.data_cube: Optional[np.ndarray] = None

        # 大气校正线程句柄
        self._atm_thread: Optional[QThread] = None
        self._atm_worker: Optional[_AtmosWorker] = None

        self._init_ui()

        # 订阅数据变化：当 DataManager 中的数据更新时，同步缓存
        self.data_manager.data_changed.connect(self.on_data_changed)

    # ---------- UI 构建 ----------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # === 辐射定标组 ===
        calib_group = QGroupBox("辐射定标")
        calib_layout = QVBoxLayout(calib_group)

        self.chk_auto_calib = QCheckBox("启用自动转换（按最大值归一化到 0-1）")
        self.chk_auto_calib.setChecked(True)
        calib_layout.addWidget(self.chk_auto_calib)

        form = QFormLayout()
        self.edit_gain = QLineEdit("1.0")
        self.edit_offset = QLineEdit("0.0")
        form.addRow("手动增益 (Gain)：", self.edit_gain)
        form.addRow("手动偏移 (Offset)：", self.edit_offset)
        calib_layout.addLayout(form)

        # 当启用自动转换时，禁用手动输入框
        self.chk_auto_calib.toggled.connect(self._on_auto_calib_toggled)
        self._on_auto_calib_toggled(self.chk_auto_calib.isChecked())

        btn_apply_calib = QPushButton("应用定标")
        btn_apply_calib.clicked.connect(self.apply_radiometric_calibration)
        calib_layout.addWidget(btn_apply_calib)

        layout.addWidget(calib_group)

        # === 大气校正组 ===
        atm_group = QGroupBox("简化大气校正（基于 Py6S 概念）")
        atm_layout = QFormLayout(atm_group)

        self.combo_atm_profile = QComboBox()
        self.combo_atm_profile.addItems(
            [
                "Tropical",
                "Midlatitude Summer",
                "Midlatitude Winter",
                "Subarctic Summer",
                "Subarctic Winter",
                "US Standard 1962",
            ]
        )

        self.combo_aero_profile = QComboBox()
        self.combo_aero_profile.addItems(
            [
                "Rural",
                "Urban",
                "Maritime",
                "Desert",
                "Biomass Burning",
                "Stratospheric",
            ]
        )

        atm_layout.addRow("大气模式：", self.combo_atm_profile)
        atm_layout.addRow("气溶胶类型：", self.combo_aero_profile)

        btn_run_atm = QPushButton("执行大气校正")
        btn_run_atm.clicked.connect(self.run_atmospheric_correction)
        from PyQt5.QtWidgets import QHBoxLayout

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_run_atm)
        atm_layout.addRow(btn_layout)

        layout.addWidget(atm_group)

        # === 进度与日志 ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log, 1)

        layout.addStretch(0)

    # ---------- 数据同步 ----------
    def on_data_changed(self, cube: np.ndarray) -> None:
        """
        当 DataManager 中的数据被其他模块更新时调用。
        这里只是缓存一个引用，实际操作在各个处理函数中进行。
        """
        self.data_cube = cube
        h, w, b = cube.shape
        self._append_log(f"接收到新数据：形状 (H, W, B) = ({h}, {w}, {b})")

    # ---------- 日志工具 ----------
    def _append_log(self, message: str, color: Optional[QColor] = None) -> None:
        """
        在日志窗口中追加一行文字。
        若提供 color，则使用该颜色显示（例如错误信息用红色）。
        """
        if color is not None:
            self.text_log.setTextColor(color)
        else:
            # 使用默认颜色
            self.text_log.setTextColor(QColor(0, 0, 0))
        self.text_log.append(message)

    def _on_auto_calib_toggled(self, checked: bool) -> None:
        """
        当“启用自动转换”勾选状态变化时，启用/禁用手动增益与偏移输入框。
        """
        self.edit_gain.setEnabled(not checked)
        self.edit_offset.setEnabled(not checked)

    # ---------- 辐射定标逻辑 ----------
    def apply_radiometric_calibration(self) -> None:
        """
        应用简单的辐射定标：
        - 若启用自动转换：按每个波段的最大值归一化到 [0, 1];
        - 否则按用户输入的 Gain / Offset 做线性变换：R = Gain * DN + Offset。

        结果通过 DataManager 回写并通知其他模块（尤其是可视化模块）刷新。
        """
        if self.data_cube is None:
            QMessageBox.warning(self, "未加载数据", "请先在“数据读取”模块中加载数据。")
            return

        cube = self.data_cube.astype(np.float32, copy=True)

        if self.chk_auto_calib.isChecked():
            self._append_log("执行自动辐射定标：按最大值归一化到 [0, 1]...")
            # 针对整个立方体按全局最大值归一化，也可以按波段分开归一化
            max_val = float(np.max(cube))
            if max_val <= 0:
                self._append_log("数据最大值 <= 0，无法进行归一化。", QColor(200, 0, 0))
                return
            cube /= max_val
        else:
            try:
                gain = float(self.edit_gain.text().strip())
                offset = float(self.edit_offset.text().strip())
            except ValueError:
                QMessageBox.warning(self, "参数错误", "请正确输入增益和偏移值（浮点数）。")
                return
            self._append_log(f"执行手动辐射定标：R = {gain} * DN + {offset}")
            cube = gain * cube + offset

        # 将结果写回 DataManager，并更新本地引用
        self.data_manager.set_data(cube)
        self.data_cube = cube
        self._append_log("辐射定标完成，数据已更新。")

    # ---------- 大气校正逻辑（异步） ----------
    def run_atmospheric_correction(self) -> None:
        """
        启动大气校正工作线程，避免长时间计算阻塞 UI。
        """
        if self.data_cube is None:
            QMessageBox.warning(self, "未加载数据", "请先在“数据读取”模块中加载数据。")
            return
        if self._atm_thread is not None:
            QMessageBox.information(self, "正在处理", "已有大气校正任务在进行中，请稍候。")
            return

        atm_profile = self.combo_atm_profile.currentText()
        aero_profile = self.combo_aero_profile.currentText()

        self._append_log(f"准备执行大气校正：大气模式={atm_profile}，气溶胶类型={aero_profile}")

        # 进度条初始化
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # 创建线程与工作对象
        self._atm_thread = QThread(self)
        self._atm_worker = _AtmosWorker(self.data_cube, atm_profile, aero_profile)
        self._atm_worker.moveToThread(self._atm_thread)

        # 线程/工作对象信号连接
        self._atm_thread.started.connect(self._atm_worker.run)
        self._atm_worker.progress.connect(self.on_atmos_progress)
        self._atm_worker.log.connect(self.on_atmos_log)
        self._atm_worker.finished.connect(self.on_atmos_finished)

        # 线程结束时清理资源
        self._atm_worker.finished.connect(self._atm_thread.quit)
        self._atm_thread.finished.connect(self._atm_worker.deleteLater)
        self._atm_thread.finished.connect(self._atm_thread.deleteLater)

        # 线程真正结束后，把句柄清空，方便下次再次启动
        def _cleanup() -> None:
            self._atm_thread = None
            self._atm_worker = None

        self._atm_thread.finished.connect(_cleanup)

        # 启动线程
        self._atm_thread.start()

    # 大气校正进度与结果回调
    def on_atmos_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def on_atmos_log(self, message: str) -> None:
        self._append_log(message)

    def on_atmos_finished(self, success: bool, result: Any) -> None:
        self.progress_bar.setVisible(False)

        if not success or result is None:
            # 错误信息已经在 worker 中通过 log 发出，这里只补充一条红色提醒
            self._append_log("大气校正失败，请检查日志信息。", QColor(200, 0, 0))
            return

        # 更新数据并通知其他模块
        corrected_cube = np.asarray(result, dtype=np.float32)
        self.data_manager.set_data(corrected_cube)
        self.data_cube = corrected_cube
        self._append_log("大气校正成功，数据已更新并同步到可视化模块。")


# ---------- 页面：导出（占位） ----------
class ExportPage(QWidget):
    """
    导出模块：
    - 将当前高光谱数据立方体导出为 ENVI / CSV（GeoTIFF 入口预留）；
    - 支持导出全图或当前选中像素的光谱曲线。
    """

    def __init__(self, data_manager: DataManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data_manager = data_manager

        # 缓存的当前数据与光谱
        self.data_cube: Optional[np.ndarray] = None
        self.current_spectrum: Optional[np.ndarray] = None

        self._init_ui()

        # 订阅数据与光谱变化
        self.data_manager.data_changed.connect(self.on_data_changed)
        self.data_manager.spectrum_changed.connect(self.on_spectrum_changed)

    # ---------- UI 构建 ----------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # === 导出格式 ===
        fmt_group = QGroupBox("导出格式")
        fmt_layout = QVBoxLayout(fmt_group)
        self.chk_envi = QCheckBox("ENVI（.hdr + .dat）")
        self.chk_envi.setChecked(True)
        self.chk_csv = QCheckBox("CSV")
        self.chk_geotiff = QCheckBox("GeoTIFF（预留，暂未实现）")
        self.chk_geotiff.setEnabled(False)
        fmt_layout.addWidget(self.chk_envi)
        fmt_layout.addWidget(self.chk_csv)
        fmt_layout.addWidget(self.chk_geotiff)
        layout.addWidget(fmt_group)

        # === 范围选择 ===
        range_group = QGroupBox("导出范围")
        range_layout = QVBoxLayout(range_group)
        self.radio_full = QRadioButton("全图数据（展平成 Pixels x Bands）")
        self.radio_pixel = QRadioButton("当前选中像素的光谱曲线")
        self.radio_full.setChecked(True)
        range_layout.addWidget(self.radio_full)
        range_layout.addWidget(self.radio_pixel)

        self.range_group_btn = QButtonGroup(self)
        self.range_group_btn.addButton(self.radio_full)
        self.range_group_btn.addButton(self.radio_pixel)

        layout.addWidget(range_group)

        # === 路径选择 ===
        path_group = QGroupBox("输出路径")
        path_layout = QHBoxLayout(path_group)
        self.edit_path = QLineEdit()
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self.browse_output)
        path_layout.addWidget(self.edit_path, 1)
        path_layout.addWidget(btn_browse)
        layout.addWidget(path_group)

        # === 转换工具入口（复用 ConvertDialog） ===
        btn_convert = QPushButton("打开“数据格式转换为 ENVI”工具...")
        btn_convert.clicked.connect(self.open_convert_dialog)
        layout.addWidget(btn_convert)

        # === 进度 & 日志 ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log, 1)

        # === 执行按钮 ===
        btn_export = QPushButton("开始导出")
        btn_export.setMinimumHeight(40)
        btn_export.clicked.connect(self.start_export)
        layout.addWidget(btn_export)

    # ---------- 数据同步 ----------
    def on_data_changed(self, cube: np.ndarray) -> None:
        self.data_cube = cube
        h, w, b = cube.shape
        self._append_log(f"接收到新数据可供导出：形状 (H, W, B) = ({h}, {w}, {b})")

    def on_spectrum_changed(self, spectrum: np.ndarray) -> None:
        self.current_spectrum = spectrum
        self._append_log("已更新当前选中像素的光谱曲线。")

    # ---------- 日志工具 ----------
    def _append_log(self, message: str, color: Optional[QColor] = None) -> None:
        if color is not None:
            self.text_log.setTextColor(color)
        else:
            self.text_log.setTextColor(QColor(0, 0, 0))
        self.text_log.append(message)

    # ---------- 路径选择 ----------
    def browse_output(self) -> None:
        """
        让用户选择输出主名（不含扩展名）。
        例如用户选择 xxx.envi，内部会截掉扩展名，只保留 stem。
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出文件路径（将根据所选格式自动添加扩展名）",
            "",
            "所有文件 (*.*)",
        )
        if not path:
            return
        stem, _ = os.path.splitext(path)
        self.edit_path.setText(stem)

    # ---------- 导出入口 ----------
    def start_export(self) -> None:
        """
        校验参数并执行导出逻辑。
        当前为同步导出，数据量较大时可进一步改为 QThread。
        """
        if self.data_cube is None:
            QMessageBox.warning(self, "未加载数据", "当前没有可导出的高光谱数据。")
            return

        if not (self.chk_envi.isChecked() or self.chk_csv.isChecked() or self.chk_geotiff.isChecked()):
            QMessageBox.warning(self, "未选择格式", "请至少选择一种导出格式。")
            return

        output_stem = self.edit_path.text().strip()
        if not output_stem:
            QMessageBox.warning(self, "路径未填写", "请先选择输出路径。")
            return

        # 初始化进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._append_log("开始导出...")

        try:
            # 1. ENVI 导出
            if self.chk_envi.isChecked():
                self._append_log("导出 ENVI 格式...")
                self._export_envi(output_stem)
                self.progress_bar.setValue(40)

            # 2. CSV 导出
            if self.chk_csv.isChecked():
                self._append_log("导出 CSV 格式...")
                self._export_csv(output_stem)
                self.progress_bar.setValue(80)

            # 3. GeoTIFF（留空，预留位置）
            if self.chk_geotiff.isChecked():
                self._append_log("GeoTIFF 导出暂未实现。", QColor(150, 100, 0))

            self.progress_bar.setValue(100)
            QMessageBox.information(self, "导出完成", f"数据已成功导出到：\n{output_stem}.*")
        except Exception as e:
            self._append_log(f"导出失败：{e}", QColor(200, 0, 0))
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误：\n{e}")
        finally:
            self.progress_bar.setVisible(False)

    # ---------- 具体导出实现 ----------
    def _export_envi(self, output_stem: str) -> None:
        """
        使用 envi_writer 将当前数据立方体导出为 ENVI (.hdr + .dat)。
        仅支持全图导出。
        """
        from envi_writer import write_envi

        if self.data_cube is None:
            raise RuntimeError("没有可导出的数据立方体。")

        # 这里固定使用 BSQ 以兼容性最佳
        hdr_path, dat_path = write_envi(output_stem, self.data_cube, interleave="bsq")
        self._append_log(f"ENVI 导出完成：\n  {hdr_path}\n  {dat_path}")

    def _export_csv(self, output_stem: str) -> None:
        """
        导出 CSV：
        - 全图模式：展平为 (Pixels, Bands) 矩阵；
        - 像素模式：输出两列：Index/Wavelength 与 Value。
        """
        if self.data_cube is None:
            raise RuntimeError("没有可导出的数据立方体。")

        import numpy as np

        csv_path = output_stem + ".csv"

        if self.radio_full.isChecked():
            h, w, b = self.data_cube.shape
            pixels = h * w
            arr = self.data_cube.reshape((pixels, b))
            header = ",".join([f"Band_{i}" for i in range(b)])
            np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
            self._append_log(f"CSV 全图导出完成：{csv_path}")
        else:
            # 单像素光谱
            if self.current_spectrum is None:
                raise RuntimeError("当前未选中任何像素，无法导出光谱曲线。")

            y = np.asarray(self.current_spectrum, dtype=float).reshape(-1)
            b = y.shape[0]

            # 优先使用 DataManager 中的波长信息；否则用 band index 代替
            wl = self.data_manager.wavelengths
            if wl is not None and len(wl) == b:
                x = np.asarray(wl, dtype=float).reshape(-1)
                header = "Wavelength,Value"
            else:
                x = np.arange(b, dtype=float)
                header = "BandIndex,Value"

            arr = np.column_stack([x, y])
            np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
            self._append_log(f"CSV 光谱曲线导出完成：{csv_path}")

    # ---------- 转换工具入口 ----------
    def open_convert_dialog(self) -> None:
        if ConvertDialog is None:
            QMessageBox.critical(
                self,
                "功能不可用",
                "ConvertDialog 未能导入，请检查 convert_dialog.py 是否存在且无语法错误。",
            )
            return

        dlg = ConvertDialog(self)
        dlg.exec_()


# ---------- 页面：可视化 ----------
class VisualizationPage(QWidget):
    """
    可视化页面：显示单波段灰度图 + 像素光谱曲线，
    并提供简单的对比度拉伸控制。
    """

    def __init__(self, data_manager: DataManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.data_manager = data_manager

        self.data: Optional[np.ndarray] = None  # (H, W, B)
        self.current_band: int = 0
        self.stretch_min: float = 2.0
        self.stretch_max: float = 98.0
        self.cbar = None
        self.img_artist = None

        self._init_ui()

        # 订阅数据变化
        self.data_manager.data_changed.connect(self.on_data_changed)

    def _init_ui(self) -> None:
        layout = QGridLayout(self)

        # Matplotlib 画布
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas, 0, 0, 4, 1)

        # 提前设置光谱坐标轴标签与位置
        self.canvas.ax_spec.set_xlabel("波段索引")
        self.canvas.ax_spec.set_ylabel("反射率 / DN")
        self.canvas.ax_spec.yaxis.set_label_coords(-0.10, 0.10)
        # 初始占位
        self.canvas.ax_img.axis("off")
        self.canvas.ax_img.text(
            0.5,
            0.5,
            "未加载图像数据",
            transform=self.canvas.ax_img.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        self.canvas.ax_spec.text(
            0.5,
            0.5,
            "未加载光谱\n请先在“数据读取”模块中加载数据，\n再在左侧图像中点击像素。",
            transform=self.canvas.ax_spec.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

        # 控件区
        controls = QVBoxLayout()
        layout.addLayout(controls, 0, 1, 4, 1)

        self.info_label = QLabel("未加载数据")
        self.info_label.setWordWrap(True)
        controls.addWidget(self.info_label)

        # 波段选择
        controls.addWidget(QLabel("波段选择："))
        self.band_combo = QComboBox()
        self.band_combo.currentIndexChanged.connect(self.on_band_changed)
        controls.addWidget(self.band_combo)

        # 对比度拉伸
        controls.addWidget(QLabel("下限百分位（min %）："))
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(0)
        self.slider_min.setMaximum(50)
        self.slider_min.setValue(2)
        self.slider_min.valueChanged.connect(self.on_stretch_changed)
        controls.addWidget(self.slider_min)

        controls.addWidget(QLabel("上限百分位（max %）："))
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setMinimum(50)
        self.slider_max.setMaximum(100)
        self.slider_max.setValue(98)
        self.slider_max.valueChanged.connect(self.on_stretch_changed)
        controls.addWidget(self.slider_max)

        self.stretch_label = QLabel("对比度拉伸：2% - 98%")
        controls.addWidget(self.stretch_label)

        controls.addStretch(1)

        # 鼠标点击事件（用于光谱曲线）
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # ---------- 数据 & 显示 ----------
    def on_data_changed(self, arr: np.ndarray) -> None:
        self.data = arr
        h, w, b = arr.shape
        self.info_label.setText(f"数据已加载：形状 (H, W, B) = ({h}, {w}, {b})")

        self.band_combo.blockSignals(True)
        self.band_combo.clear()
        for i in range(b):
            self.band_combo.addItem(f"Band {i}")
        self.band_combo.setCurrentIndex(0)
        self.band_combo.blockSignals(False)

        self.current_band = 0
        self.img_artist = None
        self.cbar = None
        self.update_image()

    def apply_contrast_stretch(self, band_img: np.ndarray) -> tuple[np.ndarray, float, float]:
        """使用百分位进行简单线性拉伸。"""
        p_min = np.percentile(band_img, self.stretch_min)
        p_max = np.percentile(band_img, self.stretch_max)
        if p_max <= p_min:
            p_max = p_min + 1e-6

        img_clip = np.clip(band_img, p_min, p_max)
        return img_clip, float(p_min), float(p_max)

    def update_image(self) -> None:
        if self.data is None:
            return

        band_img = self.data[:, :, self.current_band]
        img_clip, v_min, v_max = self.apply_contrast_stretch(band_img)

        if self.img_artist is None:
            self.canvas.ax_img.cla()
            self.img_artist = self.canvas.ax_img.imshow(
                img_clip, cmap="gray", origin="upper", vmin=v_min, vmax=v_max
            )
            self.canvas.ax_img.axis("off")
            self.cbar = self.canvas.fig.colorbar(
                self.img_artist, ax=self.canvas.ax_img, fraction=0.046, pad=0.04
            )
        else:
            self.img_artist.set_data(img_clip)
            self.img_artist.set_clim(vmin=v_min, vmax=v_max)
            if self.cbar is not None:
                self.cbar.update_normal(self.img_artist)

        self.canvas.ax_img.set_title(f"Band {self.current_band}")
        self.canvas.draw()

    def update_spectrum(self, row: int, col: int) -> None:
        if self.data is None:
            return
        h, w, b = self.data.shape
        if not (0 <= row < h and 0 <= col < w):
            return

        spectrum = self.data[row, col, :]

        self.canvas.ax_spec.cla()
        self.canvas.ax_spec.plot(np.arange(b), spectrum, marker="o", markersize=3)
        self.canvas.ax_spec.set_xlabel("波段索引")
        self.canvas.ax_spec.set_ylabel("反射率 / DN")
        self.canvas.ax_spec.yaxis.set_label_coords(-0.10, 0.10)
        self.canvas.ax_spec.set_title(f"光谱曲线 @ (row={row}, col={col})")
        self.canvas.ax_spec.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw()

    # ---------- 事件响应 ----------
    def on_band_changed(self, index: int) -> None:
        if self.data is None:
            return
        self.current_band = int(index)
        self.update_image()

    def on_stretch_changed(self) -> None:
        v_min = self.slider_min.value()
        v_max = self.slider_max.value()
        if v_min >= v_max:
            v_min = min(v_max - 1, v_min)
            self.slider_min.blockSignals(True)
            self.slider_min.setValue(v_min)
            self.slider_min.blockSignals(False)
        self.stretch_min = float(v_min)
        self.stretch_max = float(v_max)

        self.stretch_label.setText(
            f"对比度拉伸：{self.stretch_min:.0f}% - {self.stretch_max:.0f}%"
        )
        self.update_image()

    def on_click(self, event: Any) -> None:
        """在图像上点击，显示该像素位置的光谱曲线。"""
        if self.data is None:
            return
        if event.inaxes != self.canvas.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        self.update_spectrum(row, col)
        # 将当前像素的光谱曲线同步到 DataManager，供导出模块使用
        spectrum = self.data[row, col, :]
        self.data_manager.set_current_spectrum(spectrum)

