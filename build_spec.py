# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置 - 高光谱数据处理平台
# 打包命令（在项目根目录执行）：
#   pyinstaller build_spec.py
# 或单目录模式便于调试：
#   pyinstaller --onedir build_spec.py
# 说明：使用 .py 扩展名可避免编辑器将 .spec 误识别为 RPM 规格文件。

# 资源文件：样式表（运行 pyinstaller 时请在项目根目录执行）
datas = [
    ('style.qss', '.'),
    ('style_dark.qss', '.'),
    ('assets/hyspec_background.png', 'assets'),
]
# 若有 icons 目录：datas.append(('icons', 'icons'))

# 隐藏导入：数据读取与预处理可能用到的库
hiddenimports = [
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'spectral',           # ENVI 等格式
    'h5py',               # HDF5
    'cv2',                # 部分预处理可能用到（若未用可删）
]
# 若使用 py6s 做大气校正，取消下一行注释：
# hiddenimports.append('Py6S')

a = Analysis(
    ['main_window.py'],   # 入口：四模块主窗口
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HyperspectralPlatform',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
