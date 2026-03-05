"""
基于已有的 nmi_channel_selector.pkl，在验证集上重新计算 NMI 曲线并画图。
不需要重新跑通道选择，只需要 pkl + OriginalModel + val 数据。

使用方法:
    conda activate Retouch
    python Re_nmi.py
"""

import os
import sys
import pickle

# 将上级目录加入搜索路径，以便导入 EfficientNet_B0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from EfficientNet_B0 import EfficientNetB0, FeatureExtractor


# ========== 配置 ==========
PKL_PATH = "./nmi_channel_selector.pkl"
MODEL_PATH = "./results/OriginalModel.pth"
VAL_DIR = "./dataset/ali/val"
TEST_DIR = "./dataset/ali/test"
MEGVII_TEST_DIR = "./dataset/megvii/test"
SAVE_PATH = "./results/nmi_curve.png"
CHECKPOINTS = [32, 64, 128, 256]
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 2


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_val_features():
    """在验证集上提取 1280 维特征。"""
    return _extract_features(VAL_DIR, "验证集")


def extract_test_features():
    """在 ali 测试集上提取 1280 维特征。"""
    return _extract_features(TEST_DIR, "Ali 测试集")


def extract_megvii_test_features():
    """在 megvii 测试集上提取 1280 维特征（跨域评估）。"""
    return _extract_features(MEGVII_TEST_DIR, "Megvii 测试集")


def _extract_features(data_dir, name):
    """通用特征提取函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    extractor = FeatureExtractor(model, pool=True, flatten=False)

    dataset = datasets.ImageFolder(data_dir, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_features = []
    all_labels = []

    print(f"提取{name}特征...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{name} Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"{name}特征 shape: {X.shape}")
    return X, y


def compute_subspace_kmeans_nmi(features_subset, labels, n_clusters, random_state=42):
    """计算特征子空间的 KMeans-NMI。"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_subset)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3,
    )
    cluster_ids = kmeans.fit_predict(features_scaled)
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    return float(nmi)


def plot_nmi_curve(val_nmi_curve, test_nmi_curve, checkpoints, save_path,
                   original_nmi_val=None, original_nmi_test=None,
                   megvii_test_nmi_curve=None, original_nmi_megvii=None):
    """画验证集、Ali测试集、Megvii测试集三折线 NMI 曲线图。"""
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = list(range(1, len(val_nmi_curve) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(x, val_nmi_curve, color="skyblue", linestyle="-", linewidth=2, label="Ali Val NMI")
    plt.plot(x, test_nmi_curve, color="orange", linestyle="-", linewidth=2, label="Ali Test NMI")

    if megvii_test_nmi_curve is not None:
        x_meg = list(range(1, len(megvii_test_nmi_curve) + 1))
        plt.plot(x_meg, megvii_test_nmi_curve, color="tomato", linestyle="-", linewidth=2, label="Megvii Test NMI")

    # OriginalModel 全 1280 通道的 NMI，标注在 x=-1 处
    if original_nmi_val is not None and original_nmi_test is not None:
        plt.scatter([-1], [original_nmi_val], color="skyblue", edgecolors="navy",
                    s=150, zorder=5, marker="D",
                    label=f"OriginalModel Val NMI={original_nmi_val:.4f}")
        plt.scatter([-1], [original_nmi_test], color="orange", edgecolors="darkorange",
                    s=150, zorder=5, marker="D",
                    label=f"OriginalModel Ali Test NMI={original_nmi_test:.4f}")
        ann_text = f"k=1280\nVal={original_nmi_val:.4f}\nTest={original_nmi_test:.4f}"
        ann_y = max(original_nmi_val, original_nmi_test)
        if original_nmi_megvii is not None:
            plt.scatter([-1], [original_nmi_megvii], color="tomato", edgecolors="darkred",
                        s=150, zorder=5, marker="D",
                        label=f"OriginalModel Megvii Test NMI={original_nmi_megvii:.4f}")
            ann_text += f"\nMegvii={original_nmi_megvii:.4f}"
            ann_y = max(ann_y, original_nmi_megvii)
        plt.annotate(
            ann_text,
            xy=(-1, ann_y),
            xytext=(10, ann_y),
            fontsize=7, color="green",
        )

    for cp in checkpoints:
        if cp <= len(val_nmi_curve):
            plt.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
            plt.scatter([cp], [val_nmi_curve[cp - 1]], color="skyblue",
                        edgecolors="navy", s=100, zorder=5)
            plt.scatter([cp], [test_nmi_curve[cp - 1]], color="orange",
                        edgecolors="darkorange", s=100, zorder=5)
            ann_text = f"k={cp}\nVal={val_nmi_curve[cp - 1]:.4f}\nTest={test_nmi_curve[cp - 1]:.4f}"
            ann_y = max(val_nmi_curve[cp - 1], test_nmi_curve[cp - 1])
            if megvii_test_nmi_curve is not None and cp <= len(megvii_test_nmi_curve):
                plt.scatter([cp], [megvii_test_nmi_curve[cp - 1]], color="tomato",
                            edgecolors="darkred", s=100, zorder=5)
                ann_text += f"\nMegvii={megvii_test_nmi_curve[cp - 1]:.4f}"
                ann_y = max(ann_y, megvii_test_nmi_curve[cp - 1])
            plt.annotate(
                ann_text,
                xy=(cp, ann_y),
                xytext=(cp + 5, ann_y),
                fontsize=7,
            )

    plt.xlabel("Number of Selected Channels", fontsize=12)
    plt.ylabel("KMeans-NMI", fontsize=12)
    plt.title("NMI vs Number of Selected Channels (Val & Test & Megvii)", fontsize=14)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"NMI 曲线图已保存到: {save_path}")


def main():
    # 加载 pkl
    if not os.path.exists(PKL_PATH):
        print(f"文件不存在: {PKL_PATH}")
        return
    with open(PKL_PATH, "rb") as f:
        selector = pickle.load(f)

    selected_indices = selector["selected_indices"]
    print(f"已加载通道选择器，共 {len(selected_indices)} 个通道")

    # 提取各数据集特征
    val_features, val_labels = extract_val_features()
    test_features, test_labels = extract_test_features()
    meg_features, meg_labels = extract_megvii_test_features()
    n_clusters_val = int(np.unique(val_labels).size)
    n_clusters_test = int(np.unique(test_labels).size)
    n_clusters_meg = int(np.unique(meg_labels).size)

    # 逐步计算 Ali 验证集 NMI 曲线
    print(f"\n在 Ali 验证集上计算 NMI 曲线 (k=1~{len(selected_indices)})...")
    val_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="Val NMI Curve"):
        subset_idx = selected_indices[:step]
        val_subset = val_features[:, subset_idx]
        nmi = compute_subspace_kmeans_nmi(val_subset, val_labels, n_clusters_val)
        val_nmi_curve.append(nmi)

    # 逐步计算 Ali 测试集 NMI 曲线
    print(f"\n在 Ali 测试集上计算 NMI 曲线 (k=1~{len(selected_indices)})...")
    test_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="Ali Test NMI Curve"):
        subset_idx = selected_indices[:step]
        test_subset = test_features[:, subset_idx]
        nmi = compute_subspace_kmeans_nmi(test_subset, test_labels, n_clusters_test)
        test_nmi_curve.append(nmi)

    # 逐步计算 Megvii 测试集 NMI 曲线（跨域）
    print(f"\n在 Megvii 测试集上计算 NMI 曲线 (k=1~{len(selected_indices)})...")
    megvii_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="Megvii Test NMI Curve"):
        subset_idx = selected_indices[:step]
        meg_subset = meg_features[:, subset_idx]
        nmi = compute_subspace_kmeans_nmi(meg_subset, meg_labels, n_clusters_meg)
        megvii_nmi_curve.append(nmi)

    # 打印 checkpoint 结果
    print("\n各维数下的 NMI（Ali 验证集）:")
    for cp in CHECKPOINTS:
        if cp <= len(val_nmi_curve):
            print(f"  k={cp}: NMI={val_nmi_curve[cp - 1]:.6f}")

    print("\n各维数下的 NMI（Ali 测试集）:")
    for cp in CHECKPOINTS:
        if cp <= len(test_nmi_curve):
            print(f"  k={cp}: NMI={test_nmi_curve[cp - 1]:.6f}")

    print("\n各维数下的 NMI（Megvii 测试集，跨域）:")
    for cp in CHECKPOINTS:
        if cp <= len(megvii_nmi_curve):
            print(f"  k={cp}: NMI={megvii_nmi_curve[cp - 1]:.6f}")

    # 如果 pkl 中有训练集曲线，也打印对比
    if "nmi_curve_train" in selector:
        train_curve = selector["nmi_curve_train"]
        print("\n各维数下的 NMI（训练集，仅供对比）:")
        for cp in CHECKPOINTS:
            if cp <= len(train_curve):
                print(f"  k={cp}: NMI={train_curve[cp - 1]:.6f}")

    # 计算 OriginalModel 全 1280 通道的 NMI（基线对比）
    print("\n计算 OriginalModel 全 1280 通道 NMI...")
    original_nmi_val = compute_subspace_kmeans_nmi(val_features, val_labels, n_clusters_val)
    original_nmi_test = compute_subspace_kmeans_nmi(test_features, test_labels, n_clusters_test)
    original_nmi_megvii = compute_subspace_kmeans_nmi(meg_features, meg_labels, n_clusters_meg)
    print(f"  OriginalModel (1280ch) Ali Val:    NMI={original_nmi_val:.6f}")
    print(f"  OriginalModel (1280ch) Ali Test:   NMI={original_nmi_test:.6f}")
    print(f"  OriginalModel (1280ch) Megvii Test: NMI={original_nmi_megvii:.6f}")

    # 画图（三折线）
    plot_nmi_curve(
        val_nmi_curve, test_nmi_curve, CHECKPOINTS, SAVE_PATH,
        original_nmi_val=original_nmi_val,
        original_nmi_test=original_nmi_test,
        megvii_test_nmi_curve=megvii_nmi_curve,
        original_nmi_megvii=original_nmi_megvii,
    )

    # 更新 pkl
    selector["nmi_curve"] = val_nmi_curve
    selector["nmi_curve_test"] = test_nmi_curve
    selector["nmi_curve_megvii_test"] = megvii_nmi_curve
    selector["checkpoint_results"] = {}
    selector["checkpoint_results_test"] = {}
    selector["checkpoint_results_megvii_test"] = {}
    for cp in CHECKPOINTS:
        if cp <= len(val_nmi_curve):
            selector["checkpoint_results"][cp] = val_nmi_curve[cp - 1]
        if cp <= len(test_nmi_curve):
            selector["checkpoint_results_test"][cp] = test_nmi_curve[cp - 1]
        if cp <= len(megvii_nmi_curve):
            selector["checkpoint_results_megvii_test"][cp] = megvii_nmi_curve[cp - 1]

    with open(PKL_PATH, "wb") as f:
        pickle.dump(selector, f)
    print(f"已更新 pkl: {PKL_PATH}")


if __name__ == "__main__":
    main()
