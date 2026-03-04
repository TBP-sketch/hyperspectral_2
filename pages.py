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
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
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
    QSlider,
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

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data

    def set_data(self, arr: np.ndarray) -> None:
        self._data = arr
        self.data_changed.emit(arr)


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


# ---------- 页面：预处理（占位） ----------
class PreprocessPage(QWidget):
    """预处理模块（占位，后续可扩展为滤波、裁剪、波段选择等功能）。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("预处理模块开发中...\n\n未来将在此添加光谱预处理、噪声抑制、波段选择等功能。")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


# ---------- 页面：导出（占位） ----------
class ExportPage(QWidget):
    """导出模块（占位，后续可扩展为导出 ENVI / GeoTIFF / PNG 等）。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("导出模块开发中...\n\n未来将在此添加导出 ENVI、GeoTIFF、图像快照等功能。")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


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

