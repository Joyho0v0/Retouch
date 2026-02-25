import numpy as np
import pickle
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler



def _infer_n_clusters_from_labels(labels):
    unique = np.unique(labels)
    return int(unique.size)

def compute_nmi_for_feature_matrix(features, labels, n_clusters=None, random_state=42):
    if n_clusters is None:
        n_clusters = _infer_n_clusters_from_labels(labels)
    n_clusters = int(n_clusters)

    n_classes = _infer_n_clusters_from_labels(labels)
    if n_clusters > max(10 * n_classes, 50):
        print(
            f"[Warn] n_clusters={n_clusters} is much larger than n_classes={n_classes}. "
            "KMeans->NMI will often look artificially low with very large cluster counts. "
            "Consider setting n_clusters to n_classes (or a small multiple)."
        )

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)    # 这步做的是啥哦

    # 使用k-means将高维特征离散化为簇ID
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3
    )
    cluster_ids = kmeans.fit_predict(features_scaled)  # [B]

    # 计算nmi
    nmi = normalized_mutual_info_score(labels, cluster_ids)

    return nmi, cluster_ids

def evaluate_channel_selection_nmi(
        original_features,
        selected_features,
        labels,
    n_clusters=None,
    save_path=None,
    visualize_tsne: bool = False,
    tsne_out_dir: str = "./results/t-sne",
    tsne_n_samples: int = 3000,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
):
    effective_n_clusters = n_clusters
    if effective_n_clusters is None:
        effective_n_clusters = _infer_n_clusters_from_labels(labels)

    print("开始评估NMI")
    print(" 计算原始特征nmi")
    nmi_original, _ = compute_nmi_for_feature_matrix(
        original_features, labels, n_clusters=effective_n_clusters
    )

    # 计算筛选后的nmi
    print("筛选后的特征nmi")
    nmi_selected, _ = compute_nmi_for_feature_matrix(
        selected_features, labels, n_clusters=effective_n_clusters
    )

    # 计算保留率
    retention_rate = nmi_selected / (nmi_original + 1e-8)

    # 打印结果
    print("*"*50)
    print(f"原始特征 (1280D) NMI: {nmi_original:.6f}")
    print(f"筛选特征 (128D)  NMI: {nmi_selected:.6f}")
    print(f"NMI 保留率:          {retention_rate:.2%}")
    print("*"*50)

    result = {
        'nmi_original' : nmi_original,
        'nmi_selected' : nmi_selected,
        'retention_rate' : retention_rate,
        'n_clusters' : effective_n_clusters,
        'original_shape' : original_features.shape,
        'selected_shape' : selected_features.shape
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"结果已存至：{save_path}")

    if visualize_tsne:
        try:
            from tsne_visualize import save_tsne_comparison
            out_path = save_tsne_comparison(
                original_features=original_features,
                selected_features=selected_features,
                labels=labels,
                out_dir=tsne_out_dir,
                random_state=tsne_random_state,
                n_samples=tsne_n_samples,
                perplexity=tsne_perplexity,
            )
            result["tsne_path"] = out_path
            print(f"t-SNE 图已保存：{out_path}")
        except Exception as e:
            print(f"[Warn] t-SNE 可视化失败：{e}")
    
    return result

