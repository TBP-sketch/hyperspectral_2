# 高光谱数据处理平台（HyperspectralPlatform）

本项目是一个基于 Python + PyQt5 的桌面端高光谱数据处理软件，提供：

- 数据读取（`.npy` / ENVI `.hdr` / `.mat`）
- 预处理（辐射定标、简化大气校正）
- 可视化（单波段、RGB 假彩色、像素光谱曲线）
- 导出（ENVI / CSV / GeoTIFF）

---

## 1. 运行环境要求

- 操作系统：Windows 10/11（64 位）
- Python：建议 `3.13.x`
- 依赖库：见 `requirements.txt`

安装依赖：

```bash
py -m pip install -r requirements.txt
```

---

## 2. 默认安装与启动（开发态）

在项目根目录执行：

```bash
py main_window.py
```

启动后默认进入主界面，左侧包含四个模块导航：数据读取、预处理、导出、可视化。

---

## 3. 用户手册（典型流程）

### 3.1 数据读取

1. 进入“数据读取”模块；
2. 点击“浏览”选择文件（支持 `.npy` / `.hdr` / `.mat`）；
3. 点击“加载”，加载成功后会显示数据形状信息 `(H, W, B)`。

说明：
- `.mat` 文件会自动识别候选三维数组并尽量提取波长信息；
- ENVI 读取依赖 `envi_reader.py`。

### 3.2 预处理

进入“预处理”模块可执行：

- **辐射定标**：自动归一化或按 `Gain/Offset` 手动变换；
- **简化大气校正**：后台线程执行，界面显示进度条与日志。

处理完成后，结果会同步到可视化与导出模块。

### 3.3 可视化

进入“可视化”模块可执行：

- 单波段显示（支持波段选择与对比度拉伸）；
- RGB 假彩色合成（分别选择 R/G/B 波段）；
- 在图像中点击像素查看对应光谱曲线。

### 3.4 导出

进入“导出”模块：

1. 选择导出格式：ENVI / CSV / GeoTIFF；
2. 选择导出范围：全图或当前像素光谱；
3. 设置输出路径并点击“开始导出”。

---

## 4. 复现安装包（`.exe`）指南

### 4.1 打包前准备

1. 安装 PyInstaller：

```bash
py -m pip install pyinstaller
```

2. 确保依赖已安装：

```bash
py -m pip install -r requirements.txt
```

### 4.2 一键打包命令

本项目使用 `HyperspectralPlatform.spec` 作为正式打包配置（已包含 `rasterio` 运行时收集）：

```bash
py -m PyInstaller "HyperspectralPlatform.spec" --noconfirm --clean
```

### 4.3 产物位置

- 可执行文件：`dist/HyperspectralPlatform.exe`

---

## 5. 关键资源与配置说明

- 样式文件：`style.qss`、`style_dark.qss`
- 资源文件：`assets/hyspec_background.png`、`assets/app_logo.png`
- 打包配置：`HyperspectralPlatform.spec`、`build_spec.py`

---

## 6. 项目目录

```text
hyperspectral/
  main_window.py
  pages.py
  dialogs.py
  ui_messages.py
  app_background.py
  core/
    loader.py
  envi_reader.py
  envi_writer.py
  geotiff_writer.py
  style.qss
  style_dark.qss
  assets/
    hyspec_background.png
    app_logo.png
  requirements.txt
  HyperspectralPlatform.spec
```
