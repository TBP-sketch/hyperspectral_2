import sys
from typing import Optional

import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


matplotlib.use("Qt5Agg")

# 全局设置中文字体，防止坐标轴和标题出现方框
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False


class MplCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout()
        super().__init__(self.fig)
        self.setParent(parent)


class RawFileDialog(QDialog):
    """RAW 文件参数输入对话框"""

    def __init__(self, parent=None):
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
        self.interleave_combo.addItems(["BSQ (Band Sequential)", "BIL (Band Interleaved by Line)", "BIP (Band Interleaved by Pixel)"])
        self.interleave_combo.setCurrentText("BSQ (Band Sequential)")
        layout.addWidget(self.interleave_combo, 5, 1)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 6, 0, 1, 2)

        self.setLayout(layout)

    def get_params(self):
        """获取用户输入的参数"""
        try:
            height = int(self.edit_height.text())
            width = int(self.edit_width.text())
            bands = int(self.edit_bands.text())
            dtype_str = self.dtype_combo.currentText()
            byteorder_str = "<" if "little" in self.byteorder_combo.currentText() else ">"
            interleave = self.interleave_combo.currentText().split()[0]  # BSQ, BIL, or BIP

            # 构建完整的 dtype 字符串（包含字节顺序）
            if dtype_str.startswith("float") or dtype_str.startswith("int"):
                dtype = np.dtype(byteorder_str + dtype_str)
            else:  # uint8, uint16
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


class HyperSpectralViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高光谱图像浏览器（初版）")
        self.resize(1000, 600)

        self.data: Optional[np.ndarray] = None  # (H, W, B)
        self.current_band: int = 0
        self.stretch_min: float = 2.0
        self.stretch_max: float = 98.0 
        self.cbar = None  # 记录当前颜色条
        self.img_artist = None  # 记录当前图像句柄，用于只更新数据

        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        # matplotlib 画布
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas, 0, 0, 4, 1)
        # 提前设置光谱坐标轴的中文标签，并调整纵轴标题位置，避免与左侧颜色条重合
        self.canvas.ax_spec.set_xlabel("波段索引")
        self.canvas.ax_spec.set_ylabel("反射率 / DN")
        # 将纵坐标标题移动到纵轴下部（轴坐标系 y≈0.1，对应数据大约 0~0.2 区域）
        self.canvas.ax_spec.yaxis.set_label_coords(-0.10, 0.10)
        # 初始占位显示：提示用户加载数据和点击像素
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
            "未加载光谱\n请先加载数据，再在左侧图像中点击像素",
            transform=self.canvas.ax_spec.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

        # 控件区
        controls = QVBoxLayout()
        layout.addLayout(controls, 0, 1, 4, 1)

        btn_open = QPushButton("打开高光谱数据")
        btn_open.clicked.connect(self.open_file)
        controls.addWidget(btn_open)

        self.info_label = QLabel("未加载数据")
        self.info_label.setWordWrap(True)
        controls.addWidget(self.info_label)

        # 波段选择
        controls.addWidget(QLabel("波段选择："))
        self.band_combo = QComboBox()
        self.band_combo.currentIndexChanged.connect(self.on_band_changed)
        controls.addWidget(self.band_combo)

        # 对比度拉伸（百分位）
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

        # 连接鼠标点击事件，用于光谱曲线显示
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # ---------- 数据读取 ----------
    def open_file(self):
        """
        支持格式：
        - .npy: 直接加载为 (H, W, B)
        - .raw: 原始二进制文件，需要用户输入参数（高度、宽度、波段数、数据类型等）
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择高光谱数据文件",
            "",
            "NumPy 数组 (*.npy);;RAW 文件 (*.raw);;所有文件 (*.*)",
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith(".npy"):
                arr = np.load(file_path)
                if arr.ndim == 3:
                    self.data = arr.astype(np.float32)
                elif arr.ndim == 2:
                    raise ValueError("期望数据形状为 (H, W, B)，当前为 2 维。")
                else:
                    raise ValueError(f"不支持的数据维度：{arr.shape}")

            elif file_path.lower().endswith(".raw"):
                # RAW 文件：弹出参数对话框
                dialog = RawFileDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    params = dialog.get_params()
                    arr = self.load_raw_file(file_path, params)
                    self.data = arr.astype(np.float32)
                else:
                    return  # 用户取消了对话框

            else:
                raise ValueError("不支持的文件格式。支持格式：.npy, .raw")

        except Exception as e:
            self.info_label.setText(f"加载失败：{e}")
            self.data = None
            self.band_combo.clear()
            self.canvas.ax_img.cla()
            self.canvas.ax_spec.cla()
            self.canvas.draw()
            return

        h, w, b = self.data.shape
        self.info_label.setText(f"数据已加载：形状 (H, W, B) = ({h}, {w}, {b})")

        # 初始化波段列表
        self.band_combo.blockSignals(True)
        self.band_combo.clear()
        for i in range(b):
            self.band_combo.addItem(f"Band {i}")
        self.band_combo.setCurrentIndex(0)
        self.band_combo.blockSignals(False)

        self.current_band = 0
        self.img_artist = None  # 重置图像句柄，因为数据改变了
        self.cbar = None
        self.update_image()

    def load_raw_file(self, file_path: str, params: dict) -> np.ndarray:
        """
        读取 RAW 二进制文件

        参数:
            file_path: RAW 文件路径
            params: 参数字典，包含 height, width, bands, dtype, interleave

        返回:
            形状为 (H, W, B) 的 NumPy 数组
        """
        height = params["height"]
        width = params["width"]
        bands = params["bands"]
        dtype = params["dtype"]
        interleave = params["interleave"]

        # 读取原始二进制数据
        with open(file_path, "rb") as f:
            raw_data = np.fromfile(f, dtype=dtype)

        # 根据数据排列方式重组为 (H, W, B)
        expected_size = height * width * bands
        if raw_data.size != expected_size:
            raise ValueError(
                f"文件大小不匹配！\n"
                f"文件包含 {raw_data.size} 个元素，\n"
                f"但根据参数 (H={height}, W={width}, B={bands}) 期望 {expected_size} 个元素。"
            )

        if interleave == "BSQ":
            # Band Sequential: 每个波段连续存储，形状为 (B, H, W)
            raw_data = raw_data.reshape((bands, height, width))
            # 转换为 (H, W, B)
            arr = np.transpose(raw_data, (1, 2, 0))
        elif interleave == "BIL":
            # Band Interleaved by Line: 每行包含所有波段，形状为 (H, B, W)
            raw_data = raw_data.reshape((height, bands, width))
            # 转换为 (H, W, B)
            arr = np.transpose(raw_data, (0, 2, 1))
        elif interleave == "BIP":
            # Band Interleaved by Pixel: 每个像素的所有波段连续，形状为 (H, W, B)
            arr = raw_data.reshape((height, width, bands))
        else:
            raise ValueError(f"不支持的数据排列方式：{interleave}")

        return arr

    # ---------- 对比度拉伸 ----------
    def apply_contrast_stretch(self, band_img: np.ndarray):
        """
        使用百分位进行简单线性拉伸。

        返回值：
        - img_clip: 按百分位裁剪后的图像（仍在原始数值范围内）
        - v_min, v_max: 对应的显示下限、上限，用于更新颜色条刻度
        """
        p_min = np.percentile(band_img, self.stretch_min)
        p_max = np.percentile(band_img, self.stretch_max)
        if p_max <= p_min:
            p_max = p_min + 1e-6

        img_clip = np.clip(band_img, p_min, p_max)
        return img_clip, float(p_min), float(p_max)

    # ---------- 图像 & 光谱更新 ----------
    def update_image(self):
        if self.data is None:
            return

        band_img = self.data[:, :, self.current_band]
        img_clip, v_min, v_max = self.apply_contrast_stretch(band_img)

        if self.img_artist is None:
            # 第一次绘制：创建图像和颜色条
            self.canvas.ax_img.cla()
            self.img_artist = self.canvas.ax_img.imshow(
                img_clip, cmap="gray", origin="upper", vmin=v_min, vmax=v_max
            )
            self.canvas.ax_img.axis("off")
            self.cbar = self.canvas.fig.colorbar(
                self.img_artist, ax=self.canvas.ax_img, fraction=0.046, pad=0.04
            )
        else:
            # 后续只更新图像数据和显示范围
            self.img_artist.set_data(img_clip)
            self.img_artist.set_clim(vmin=v_min, vmax=v_max)
            if self.cbar is not None:
                # 让颜色条的刻度范围随对比度拉伸同步更新
                self.cbar.update_normal(self.img_artist)

        self.canvas.ax_img.set_title(f"Band {self.current_band}")

        self.canvas.draw()

    def update_spectrum(self, row: int, col: int):
        if self.data is None:
            return
        h, w, b = self.data.shape
        if not (0 <= row < h and 0 <= col < w):
            return

        spectrum = self.data[row, col, :]

        self.canvas.ax_spec.cla()
        self.canvas.ax_spec.plot(np.arange(b), spectrum, marker="o", markersize=3)
        # 重新设置坐标轴标签与位置（确保在清空后仍然存在）
        self.canvas.ax_spec.set_xlabel("波段索引")
        self.canvas.ax_spec.set_ylabel("反射率 / DN")
        self.canvas.ax_spec.yaxis.set_label_coords(-0.10, 0.10)
        self.canvas.ax_spec.set_title(f"光谱曲线 @ (row={row}, col={col})")
        self.canvas.ax_spec.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw()

    # ---------- 事件响应 ----------
    def on_band_changed(self, index: int):
        if self.data is None:
            return
        self.current_band = int(index)
        self.update_image()

    def on_stretch_changed(self):
        v_min = self.slider_min.value()
        v_max = self.slider_max.value()
        if v_min >= v_max:
            v_min = min(v_max - 1, v_min)
            self.slider_min.blockSignals(True)
            self.slider_min.setValue(v_min)
            self.slider_min.blockSignals(False)
        self.stretch_min = float(v_min)
        self.stretch_max = float(v_max)

        self.stretch_label.setText(f"对比度拉伸：{self.stretch_min:.0f}% - {self.stretch_max:.0f}%")
        self.update_image()

    def on_click(self, event):
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


def main():
    app = QApplication(sys.argv)
    win = HyperSpectralViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()