import os
import csv
import io
from collections import defaultdict

import matplotlib.pyplot as plt


def load_total_csv(csv_path):
    """读取 result/total.csv，返回按 k 分组的曲线数据。

    返回:
        data: dict[k] = {
            "epoch": [..],
            "train_acc": [..],
            "val_acc": [..],
        }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    data = defaultdict(lambda: {"epoch": [], "train_acc": [], "val_acc": []})

    # 直接用 csv.reader 遇到包含 NUL 的行会报错，这里先把文件读为字符串，
    # 去掉其中的 "\0" 再交给 csv.DictReader 解析。
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\0", "")

    buffer = io.StringIO(text)
    reader = csv.DictReader(buffer)
    required_cols = {"k", "epoch", "train_acc", "val_acc"}
    if not required_cols.issubset(reader.fieldnames or []):
        raise ValueError(
            f"CSV 列缺失，期望包含: {required_cols}，实际: {reader.fieldnames}"
        )

    for row in reader:
        try:
            k = int(float(row["k"]))
        except ValueError:
            # 如果 k 不是数字，就当做字符串分组
            k = row["k"]

        try:
            epoch = int(float(row["epoch"]))
            train_acc = float(row["train_acc"])
            val_acc = float(row["val_acc"])
        except ValueError:
            # 跳过无法解析的行
            continue

        data[k]["epoch"].append(epoch)
        data[k]["train_acc"].append(train_acc)
        data[k]["val_acc"].append(val_acc)

    # 按 epoch 排序
    for k, d in data.items():
        if not d["epoch"]:
            continue
        zipped = sorted(
            zip(d["epoch"], d["train_acc"], d["val_acc"]), key=lambda x: x[0]
        )
        epochs, train_accs, val_accs = zip(*zipped)
        d["epoch"] = list(epochs)
        d["train_acc"] = list(train_accs)
        d["val_acc"] = list(val_accs)

    return data


def plot_compare_curves(data, save_dir):
    """根据按 k 分组的数据画 train/val 精度对比折线图并保存。"""
    if not data:
        raise ValueError("没有可用数据，data 为空")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "k_compare_acc.png")

    plt.figure(figsize=(10, 8))
    # 为不同 k 分配不同点形状和线型，便于区分
    markers = ["o", "s", "^", "D", "x", "+"]
    linestyles = ["-", "--", "-.", ":"]

    # 上图：train_acc
    plt.subplot(2, 1, 1)
    for idx, (k, d) in enumerate(sorted(data.items(), key=lambda x: x[0])):
        label = f"k={k}"
        mk = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        plt.plot(
            d["epoch"],
            d["train_acc"],
            marker=mk,
            markersize=4,
            linewidth=1.3,
            linestyle=ls,
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Epoch for different k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    # 下图：val_acc
    plt.subplot(2, 1, 2)
    for idx, (k, d) in enumerate(sorted(data.items(), key=lambda x: x[0])):
        label = f"k={k}"
        mk = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        plt.plot(
            d["epoch"],
            d["val_acc"],
            marker=mk,
            markersize=4,
            linewidth=1.2,
            linestyle=ls,
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Validation Accuracy vs Epoch for different k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"对比折线图已保存到: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total_csv = os.path.join(base_dir, "result", "total.csv")
    save_dir = os.path.join(base_dir, "result")

    data = load_total_csv(total_csv)
    plot_compare_curves(data, save_dir)


if __name__ == "__main__":
    main()
