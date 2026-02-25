import numpy as np
import os

# ==============================
# 🔧 显式设置你的 .npy 文件路径
# ==============================
NPY_FILE_PATH = "/media/dongli911/Software/ZhuYunHao/Retouch/results/labels.npy"
# NPY_FILE_PATH = "/home/yourname/data/my_array.npy"            # Linux/macOS 示例

# 输出 TXT 文件路径（自动替换扩展名）
TXT_FILE_PATH = os.path.splitext(NPY_FILE_PATH)[0] + ".txt"

# ==============================
# 配置保存选项（按需修改）
# ==============================
DELIMITER = ','          # 分隔符：',' 或 ' ' 或 '\t'
FORMAT = '%.6f'          # 数值格式：%d（整数）、%.4f（4位小数）、%g（自动）
FLATTEN_IF_NEEDED = True # 若为3D+数组，是否自动展平？

# ==============================
# 执行转换
# ==============================

def npy_to_txt(npy_path, txt_path, delimiter=',', fmt='%.6f', flatten=True):
    if not os.path.exists(npy_path):
        print(f"❌ .npy 文件不存在: {npy_path}")
        return

    try:
        data = np.load(npy_path)
        print(f"✅ 加载成功: shape={data.shape}, dtype={data.dtype}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 处理高维数组（>2D）
    original_shape = data.shape
    if data.ndim > 2:
        if flatten:
            print(f"⚠️  检测到 {data.ndim}D 数组 {original_shape}，正在展平为 2D...")
            # 展平策略：保留最后一维，前面合并（常见于 (N, H, W) → (N*H, W)）
            if data.ndim == 3:
                data = data.reshape(-1, data.shape[-1])
            else:
                data = data.reshape(data.shape[0], -1)  # 或直接 data.flatten()[:, None]
            print(f"   新形状: {data.shape}")
        else:
            print("❌ 不支持保存 3D 及以上数组到 TXT（除非展平）。")
            return

    # 保存为 TXT
    try:
        np.savetxt(txt_path, data, delimiter=delimiter, fmt=fmt)
        print(f"💾 成功保存 TXT 文件至: {txt_path}")
        print(f"   格式: 分隔符='{delimiter}', 数值格式='{fmt}'")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ==============================
# 运行主程序
# ==============================
if __name__ == "__main__":
    npy_to_txt(
        npy_path=NPY_FILE_PATH,
        txt_path=TXT_FILE_PATH,
        delimiter=DELIMITER,
        fmt=FORMAT,
        flatten=FLATTEN_IF_NEEDED
    )