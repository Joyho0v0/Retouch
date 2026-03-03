"""
基于已有的 nmi_channel_selector.pkl，直接在测试集上提取特征并画各 k 值的 t-SNE 图。
不需要训练好的通道选择模型权重，只需 pkl + OriginalModel + 数据。

输出与 result/tsne/ 下格式一致：tsne_k-1.png, tsne_k32.png, tsne_k64.png, ...

使用方法:
    conda activate Retouch
    python re_tsne.py
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from EfficientNet_B0 import EfficientNetB0, FeatureExtractor


# ========== 配置 ==========
PKL_PATH = "./nmi_channel_selector.pkl"
MODEL_PATH = "./results/OriginalModel.pth"
DATA_DIR = "./dataset/megvii/test"
OUT_DIR = "./result/tsne_test_megvii"
CHECKPOINTS = [32, 64, 128, 256]
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_SAMPLES = 2000
PERPLEXITY = 30.0
RANDOM_STATE = 42


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_features():
    """用 OriginalModel 提取全部 1280 维特征。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    extractor = FeatureExtractor(model, pool=True, flatten=False)

    dataset = datasets.ImageFolder(DATA_DIR, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_features = []
    all_labels = []

    print(f"提取特征中... ({DATA_DIR})")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"特征 shape: {X.shape}")
    return X, y


def subsample(features, labels, max_samples, seed):
    """随机下采样，避免 t-SNE 太慢。"""
    n = features.shape[0]
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return features, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], labels[idx]


def run_tsne(features, random_state, perplexity):
    """PCA 预降维 + t-SNE 降到 2D。"""
    n = features.shape[0]

    pca_dim = min(features.shape[1], 50)
    Xp = PCA(n_components=pca_dim, random_state=random_state).fit_transform(features)

    max_perp = (n - 1) / 3.0
    use_perp = min(perplexity, max_perp)
    use_perp = max(use_perp, 1.0)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=float(use_perp),
        random_state=random_state,
    )
    Z = tsne.fit_transform(Xp)
    return Z


def plot_tsne(Z, labels, title, save_path):
    """画 t-SNE 散点图。"""
    classes = np.unique(labels)
    plt.figure(figsize=(6, 5))

    for c in classes:
        mask = labels == c
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.75, label=str(c))

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", frameon=False, markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # 加载 pkl
    if not os.path.exists(PKL_PATH):
        print(f"文件不存在: {PKL_PATH}")
        return
    with open(PKL_PATH, "rb") as f:
        selector = pickle.load(f)
    selected_indices = selector["selected_indices"]
    print(f"已加载通道选择器，共 {len(selected_indices)} 个通道")

    # 提取特征
    features, labels = extract_features()

    # 创建输出目录
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # k=-1：OriginalModel 全 1280 通道
    print("\n--- t-SNE k=-1 (全 1280 通道) ---")
    feats_sub, labels_sub = subsample(features, labels, MAX_SAMPLES, RANDOM_STATE)
    Z = run_tsne(feats_sub, RANDOM_STATE, PERPLEXITY)
    save_path = os.path.join(OUT_DIR, "tsne_k-1.png")
    plot_tsne(Z, labels_sub, "t-SNE for k=-1 (Original 1280ch)", save_path)
    print(f"已保存: {save_path}")

    # 各 checkpoint k 值
    for k in CHECKPOINTS:
        if k > len(selected_indices):
            print(f"跳过 k={k}（选择器只有 {len(selected_indices)} 个通道）")
            continue

        print(f"\n--- t-SNE k={k} ---")
        indices_k = selected_indices[:k]
        feats_k = features[:, indices_k]

        feats_sub, labels_sub = subsample(feats_k, labels, MAX_SAMPLES, RANDOM_STATE)
        Z = run_tsne(feats_sub, RANDOM_STATE, PERPLEXITY)

        save_path = os.path.join(OUT_DIR, f"tsne_k{k}.png")
        plot_tsne(Z, labels_sub, f"t-SNE for k={k}", save_path)
        print(f"已保存: {save_path}")

    print(f"\n全部完成，图片保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
