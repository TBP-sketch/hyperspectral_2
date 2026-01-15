import numpy as np

data = np.random.rand(100, 100, 10).astype(np.float32)  # 100x100 像素，10 个波段
np.save("test_hsi.npy", data)