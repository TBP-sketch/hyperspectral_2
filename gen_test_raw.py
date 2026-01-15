"""
生成测试用的 RAW 文件，用于验证 RAW 文件导入功能
"""
import numpy as np


def generate_test_raw_bsq(output_path="test_hsi.raw", height=100, width=100, bands=10, dtype=np.float32):
    """
    生成 BSQ (Band Sequential) 格式的 RAW 文件

    参数:
        output_path: 输出文件路径
        height: 图像高度（行数）
        width: 图像宽度（列数）
        bands: 波段数
        dtype: 数据类型
    """
    # 生成随机数据，形状为 (H, W, B)
    data = np.random.rand(height, width, bands).astype(dtype)

    # 转换为 BSQ 格式: (B, H, W)
    data_bsq = np.transpose(data, (2, 0, 1))

    # 写入文件
    data_bsq.tofile(output_path)
    print(f"已生成 BSQ 格式 RAW 文件: {output_path}")
    print(f"  形状: (H={height}, W={width}, B={bands})")
    print(f"  数据类型: {dtype}")
    print(f"  数据排列: BSQ (Band Sequential)")
    print(f"  文件大小: {data_bsq.nbytes} 字节")


def generate_test_raw_bil(output_path="test_hsi_bil.raw", height=100, width=100, bands=10, dtype=np.float32):
    """
    生成 BIL (Band Interleaved by Line) 格式的 RAW 文件
    """
    data = np.random.rand(height, width, bands).astype(dtype)

    # 转换为 BIL 格式: (H, B, W)
    data_bil = np.transpose(data, (0, 2, 1))

    data_bil.tofile(output_path)
    print(f"已生成 BIL 格式 RAW 文件: {output_path}")
    print(f"  形状: (H={height}, W={width}, B={bands})")
    print(f"  数据类型: {dtype}")
    print(f"  数据排列: BIL (Band Interleaved by Line)")
    print(f"  文件大小: {data_bil.nbytes} 字节")


def generate_test_raw_bip(output_path="test_hsi_bip.raw", height=100, width=100, bands=10, dtype=np.float32):
    """
    生成 BIP (Band Interleaved by Pixel) 格式的 RAW 文件
    """
    data = np.random.rand(height, width, bands).astype(dtype)

    # BIP 格式已经是 (H, W, B)，直接写入
    data.tofile(output_path)
    print(f"已生成 BIP 格式 RAW 文件: {output_path}")
    print(f"  形状: (H={height}, W={width}, B={bands})")
    print(f"  数据类型: {dtype}")
    print(f"  数据排列: BIP (Band Interleaved by Pixel)")
    print(f"  文件大小: {data.nbytes} 字节")


if __name__ == "__main__":
    print("生成测试 RAW 文件...\n")

    # 生成 BSQ 格式（最常用）
    generate_test_raw_bsq("test_hsi_bsq.raw", height=100, width=100, bands=10, dtype=np.float32)
    print()

    # 生成 BIL 格式
    generate_test_raw_bil("test_hsi_bil.raw", height=100, width=100, bands=10, dtype=np.float32)
    print()

    # 生成 BIP 格式
    generate_test_raw_bip("test_hsi_bip.raw", height=100, width=100, bands=10, dtype=np.float32)
    print()

    print("所有测试文件生成完成！")
    print("\n使用方法：")
    print("1. 在软件中点击 '打开高光谱数据'")
    print("2. 选择生成的 .raw 文件")
    print("3. 在弹出的对话框中输入参数：")
    print("   - 高度: 100")
    print("   - 宽度: 100")
    print("   - 波段数: 10")
    print("   - 数据类型: float32")
    print("   - 字节顺序: little-endian (<)")
    print("   - 数据排列: 根据选择的文件选择对应的格式（BSQ/BIL/BIP）")

