"""
随机通道选择器：每次从 1280 个通道中随机选取 128 个通道，
在 ali/test 上测试 NMI，死循环搜索直到手动停止 (Ctrl+C)。
如果当前 NMI 超过历史最佳，则更新最佳通道选择器并保存。

使用方法:
    conda activate Retouch
    python RandomSelect.py

代码风格：不使用 lambda、argparse 等复杂句式，保持清晰简洁。
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from EfficientNet_B0 import EfficientNetB0, FeatureExtractor


# ========== 配置 ==========
MODEL_PATH = "./results/OriginalModel.pth"
ALI_VAL_DIR = "./dataset/ali/val"
ALI_TEST_DIR = "./dataset/ali/test"
MEGVII_TEST_DIR = "./dataset/megvii/test"
PKL_SAVE_PATH = "./random_channel_selector.pkl"
NMI_PLOT_SAVE_PATH = "./results/random_nmi_curve.png"
NUM_CHANNELS = 1280         # 原始通道总数
SELECT_K = 128              # 每次随机选取的通道数
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 2
NMI_PLOT_THRESHOLD = 0.7139  # 从最佳 NMI 超过此值开始画图


# ========== 数据预处理 ==========
def get_val_transform():
    """返回验证/测试用的图像预处理流程。"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform


# ========== 特征提取 ==========
def extract_features(data_dir, name):
    """
    加载 OriginalModel，在指定目录上提取 1280 维特征。
    
    输入:
        data_dir: 数据集目录路径（需要有 0/ 和 1/ 子目录）
        name: 数据集描述文字，用于打印
    
    返回:
        X: 特征矩阵 [N, 1280]
        y: 标签数组 [N,]
    """
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

    print(f"提取 {name} 特征...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{name} Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"{name} 特征 shape: {X.shape}")
    return X, y


# ========== NMI 计算 ==========
def compute_subspace_kmeans_nmi(features_subset, labels, n_clusters, random_state=42):
    """
    计算特征子空间的 KMeans-NMI。
    
    输入:
        features_subset: 选中通道的特征矩阵 [N, num_selected]
        labels: 真实标签 [N,]
        n_clusters: 聚类数
    
    返回:
        nmi: 归一化互信息值
    """
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


# ========== NMI 三折线图绘制 ==========
def plot_nmi_three_lines(selected_indices, val_features, val_labels,
                         test_features, test_labels,
                         meg_features, meg_labels,
                         save_path,
                         original_nmi_val=None,
                         original_nmi_test=None,
                         original_nmi_megvii=None):
    """
    给定一组选中通道索引，在三个数据集上逐步计算 NMI 曲线并画图。
    
    三条折线:
        天蓝色: Ali 验证集
        橙黄色: Ali 测试集
        番茄红: Megvii 测试集
    
    在 x=-1 处用菱形标记 OriginalModel（全 1280 通道）的 NMI 作为基线对比。
    
    输入:
        selected_indices: 选中的通道索引列表（长度 128）
        val_features: Ali 验证集特征 [N1, 1280]
        val_labels: Ali 验证集标签 [N1,]
        test_features: Ali 测试集特征 [N2, 1280]
        test_labels: Ali 测试集标签 [N2,]
        meg_features: Megvii 测试集特征 [N3, 1280]
        meg_labels: Megvii 测试集标签 [N3,]
        save_path: 图片保存路径
        original_nmi_val: OriginalModel 全 1280 通道在 Ali 验证集上的 NMI
        original_nmi_test: OriginalModel 全 1280 通道在 Ali 测试集上的 NMI
        original_nmi_megvii: OriginalModel 全 1280 通道在 Megvii 测试集上的 NMI
    """
    n_clusters_val = int(np.unique(val_labels).size)
    n_clusters_test = int(np.unique(test_labels).size)
    n_clusters_meg = int(np.unique(meg_labels).size)

    total_k = len(selected_indices)

    # 逐步计算三个数据集的 NMI 曲线
    val_nmi_curve = []
    test_nmi_curve = []
    meg_nmi_curve = []

    for step in range(1, total_k + 1):
        subset = selected_indices[:step]

        val_subset = val_features[:, subset]
        val_nmi = compute_subspace_kmeans_nmi(val_subset, val_labels, n_clusters_val)
        val_nmi_curve.append(val_nmi)

        test_subset = test_features[:, subset]
        test_nmi = compute_subspace_kmeans_nmi(test_subset, test_labels, n_clusters_test)
        test_nmi_curve.append(test_nmi)

        meg_subset = meg_features[:, subset]
        meg_nmi = compute_subspace_kmeans_nmi(meg_subset, meg_labels, n_clusters_meg)
        meg_nmi_curve.append(meg_nmi)

    # 画图
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = list(range(1, total_k + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(x, val_nmi_curve, color="skyblue", linestyle="-", linewidth=2, label="Ali Val NMI")
    plt.plot(x, test_nmi_curve, color="orange", linestyle="-", linewidth=2, label="Ali Test NMI")
    plt.plot(x, meg_nmi_curve, color="tomato", linestyle="-", linewidth=2, label="Megvii Test NMI")

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

    # 在 k=32, 64, 128 处标记
    checkpoints = [32, 64, 128]
    for cp in checkpoints:
        if cp <= total_k:
            plt.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

            # 三条曲线在 checkpoint 处的散点
            plt.scatter([cp], [val_nmi_curve[cp - 1]], color="skyblue",
                        edgecolors="navy", s=100, zorder=5)
            plt.scatter([cp], [test_nmi_curve[cp - 1]], color="orange",
                        edgecolors="darkorange", s=100, zorder=5)
            plt.scatter([cp], [meg_nmi_curve[cp - 1]], color="tomato",
                        edgecolors="darkred", s=100, zorder=5)

            # 标注文字
            ann_y = max(val_nmi_curve[cp - 1], test_nmi_curve[cp - 1], meg_nmi_curve[cp - 1])
            ann_text = (f"k={cp}\n"
                        f"Val={val_nmi_curve[cp - 1]:.4f}\n"
                        f"Test={test_nmi_curve[cp - 1]:.4f}\n"
                        f"Megvii={meg_nmi_curve[cp - 1]:.4f}")
            plt.annotate(
                ann_text,
                xy=(cp, ann_y),
                xytext=(cp + 3, ann_y),
                fontsize=7,
            )

    plt.xlabel("Number of Selected Channels", fontsize=12)
    plt.ylabel("KMeans-NMI", fontsize=12)
    plt.title("NMI vs Number of Selected Channels (Random Selection)", fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"三折线 NMI 曲线图已保存到: {save_path}")


# ========== 主循环 ==========
def main():
    # 提取三个数据集的特征（只提取一次，循环中复用）
    print("=" * 60)
    print("随机通道选择器 - 特征提取阶段")
    print("=" * 60)

    val_features, val_labels = extract_features(ALI_VAL_DIR, "Ali 验证集")
    test_features, test_labels = extract_features(ALI_TEST_DIR, "Ali 测试集")
    meg_features, meg_labels = extract_features(MEGVII_TEST_DIR, "Megvii 测试集")

    n_clusters_val = int(np.unique(val_labels).size)
    n_clusters_test = int(np.unique(test_labels).size)
    n_clusters_meg = int(np.unique(meg_labels).size)

    # 初始化最佳记录
    best_nmi = -1.0
    best_indices = None
    loop_count = 0

    # 检查是否已存在通道选择器
    if os.path.exists(PKL_SAVE_PATH):
        print(f"发现已有通道选择器: {PKL_SAVE_PATH}")
        with open(PKL_SAVE_PATH, "rb") as f:
            old_selector = pickle.load(f)
        best_indices = old_selector["selected_indices"]
        # 在当前 ali/test 特征上测试已有索引的 NMI
        old_subset = test_features[:, best_indices]
        best_nmi = compute_subspace_kmeans_nmi(old_subset, test_labels, n_clusters_test)
        old_loops = old_selector.get("total_loops", "未知")
        print(f"  已有索引在 ali/test 上的 NMI = {best_nmi:.6f}")
        print(f"  历史搜索轮数: {old_loops}")
        print(f"  将以此为基准继续搜索")
    else:
        print(f"未找到已有通道选择器，将从头开始搜索")

    print("")
    print("=" * 60)
    print("随机通道选择器 - 开始搜索")
    print(f"每次从 {NUM_CHANNELS} 个通道中随机选取 {SELECT_K} 个")
    print(f"评估数据集: ali/test")
    print(f"当前最佳 NMI: {best_nmi:.6f}" if best_nmi > 0 else "当前最佳 NMI: 无")
    print(f"画图阈值: NMI > {NMI_PLOT_THRESHOLD}")
    print("按 Ctrl+C 手动停止搜索")
    print("=" * 60)
    print("")

    # 预先计算 OriginalModel 全 1280 通道的 NMI（只算一次，循环中复用）
    print("计算 OriginalModel 全 1280 通道 NMI（基线对比）...")
    orig_nmi_val = compute_subspace_kmeans_nmi(val_features, val_labels, n_clusters_val)
    orig_nmi_test = compute_subspace_kmeans_nmi(test_features, test_labels, n_clusters_test)
    orig_nmi_meg = compute_subspace_kmeans_nmi(meg_features, meg_labels, n_clusters_meg)
    print(f"  OriginalModel (1280ch) Val={orig_nmi_val:.4f}, "
          f"Test={orig_nmi_test:.4f}, Megvii={orig_nmi_meg:.4f}")
    print("")

    # 死循环搜索
    while True:
        loop_count = loop_count + 1

        # 随机选取 128 个通道索引（从 0 到 1279 中不重复地选）
        random_indices = np.random.choice(NUM_CHANNELS, size=SELECT_K, replace=False)
        random_indices = list(random_indices)

        # 在 ali/test 上计算这 128 个通道的 NMI
        test_subset = test_features[:, random_indices]
        current_nmi = compute_subspace_kmeans_nmi(test_subset, test_labels, n_clusters_test)

        # 判断是否需要更新最佳通道选择器
        if current_nmi > best_nmi:
            best_nmi = current_nmi
            best_indices = random_indices

            print(f"[第 {loop_count} 轮] *** 更新最佳 *** NMI = {best_nmi:.6f}")

            # 保存当前最佳选择器到 pkl
            selector_data = {
                "selected_indices": best_indices,
                "best_nmi_ali_test": best_nmi,
                "total_loops": loop_count,
                "select_k": SELECT_K,
                "strategy": "random_search",
            }
            with open(PKL_SAVE_PATH, "wb") as f:
                pickle.dump(selector_data, f)

            # 如果最佳 NMI 超过阈值，画三折线图
            if best_nmi > NMI_PLOT_THRESHOLD:
                print(f"  NMI > {NMI_PLOT_THRESHOLD}，正在画三折线 NMI 图...")
                plot_nmi_three_lines(
                    selected_indices=best_indices,
                    val_features=val_features,
                    val_labels=val_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    meg_features=meg_features,
                    meg_labels=meg_labels,
                    save_path=NMI_PLOT_SAVE_PATH,
                    original_nmi_val=orig_nmi_val,
                    original_nmi_test=orig_nmi_test,
                    original_nmi_megvii=orig_nmi_meg,
                )
        else:
            # 没有超过最佳值，跳过
            print(f"[第 {loop_count} 轮] 当前 NMI = {current_nmi:.6f}，最佳 NMI = {best_nmi:.6f}")


if __name__ == "__main__":
    main()
