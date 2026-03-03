"""
基于已有的 nmi_channel_selector.pkl，在验证集上重新计算 NMI 曲线并画图。
不需要重新跑通道选择，只需要 pkl + OriginalModel + val 数据。

使用方法:
    conda activate Retouch
    python Re_nmi.py
"""

import os
import pickle

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    extractor = FeatureExtractor(model, pool=True, flatten=False)

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_features = []
    all_labels = []

    print("提取验证集特征...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"验证集特征 shape: {X.shape}")
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


def plot_nmi_curve(nmi_curve, checkpoints, save_path):
    """画 NMI 随维数变化的曲线图。"""
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = list(range(1, len(nmi_curve) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(x, nmi_curve, "b-", linewidth=2, label="Greedy KMeans-NMI (Val)")

    for cp in checkpoints:
        if cp <= len(nmi_curve):
            plt.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
            plt.scatter([cp], [nmi_curve[cp - 1]], color="r", s=100, zorder=5)
            plt.annotate(
                f"k={cp}\nNMI={nmi_curve[cp - 1]:.4f}",
                xy=(cp, nmi_curve[cp - 1]),
                xytext=(cp + 5, nmi_curve[cp - 1]),
                fontsize=9,
            )

    plt.xlabel("Number of Selected Channels", fontsize=12)
    plt.ylabel("KMeans-NMI", fontsize=12)
    plt.title("NMI vs Number of Selected Channels (Val Set)", fontsize=14)
    plt.legend(loc="lower right")
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

    # 提取验证集特征
    val_features, val_labels = extract_val_features()
    n_clusters = int(np.unique(val_labels).size)

    # 逐步计算 NMI 曲线
    print(f"\n在验证集上计算 NMI 曲线 (k=1~{len(selected_indices)})...")
    val_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="NMI Curve"):
        subset_idx = selected_indices[:step]
        val_subset = val_features[:, subset_idx]
        nmi = compute_subspace_kmeans_nmi(val_subset, val_labels, n_clusters)
        val_nmi_curve.append(nmi)

    # 打印 checkpoint 结果
    print("\n各维数下的 NMI（验证集）:")
    for cp in CHECKPOINTS:
        if cp <= len(val_nmi_curve):
            print(f"  k={cp}: NMI={val_nmi_curve[cp - 1]:.6f}")

    # 如果 pkl 中有训练集曲线，也打印对比
    if "nmi_curve_train" in selector:
        train_curve = selector["nmi_curve_train"]
        print("\n各维数下的 NMI（训练集，仅供对比）:")
        for cp in CHECKPOINTS:
            if cp <= len(train_curve):
                print(f"  k={cp}: NMI={train_curve[cp - 1]:.6f}")

    # 画图
    plot_nmi_curve(val_nmi_curve, CHECKPOINTS, SAVE_PATH)

    # 更新 pkl（把验证集 NMI 曲线也写入）
    selector["nmi_curve"] = val_nmi_curve
    selector["checkpoint_results"] = {}
    for cp in CHECKPOINTS:
        if cp <= len(val_nmi_curve):
            selector["checkpoint_results"][cp] = val_nmi_curve[cp - 1]

    with open(PKL_PATH, "wb") as f:
        pickle.dump(selector, f)
    print(f"已更新 pkl: {PKL_PATH}")


if __name__ == "__main__":
    main()
