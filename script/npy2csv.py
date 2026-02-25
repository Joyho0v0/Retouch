import numpy as np
import pandas as pd
import os

# ==============================
# 📁 文件路径显式定义在代码变量中（硬编码）
# ==============================

# 输入 .npy 文件路径（请根据你的实际文件位置修改）
INPUT_NPY_FILE = r"/media/dongli911/Software/ZhuYunHao/Retouch/results/features_1280.npy"

# 输出 .csv 文件路径（可与输入同目录，也可自定义）
OUTPUT_CSV_FILE = r"/media/dongli911/Software/ZhuYunHao/Retouch/results/features_1280.csv"

# ==============================
# 🔧 转换逻辑
# ==============================

def main():
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_NPY_FILE):
        print(f"❌ 错误: 输入文件不存在 → {INPUT_NPY_FILE}")
        return

    try:
        # 加载 .npy 文件
        data = np.load(INPUT_NPY_FILE, allow_pickle=True)
        print(f"✅ 成功加载数据，形状: {data.shape}, 类型: {data.dtype}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 处理维度
    if data.ndim == 0:
        data = np.array([[data]])
    elif data.ndim == 1:
        data = data.reshape(-1, 1)  # 转为列向量
        print("📌 一维数组已转为列向量（每行一个值）")
    elif data.ndim > 2:
        print(f"⚠️  警告: 数据维度为 {data.ndim}，将自动展平为一列。")
        data = data.flatten().reshape(-1, 1)

    try:
        # 转为 DataFrame 并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV_FILE, index=False, header=False)
        print(f"🎉 成功保存 CSV 文件 → {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

if __name__ == "__main__":
    main()