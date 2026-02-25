from train import *
from EfficientNet_B0 import FeatureExtractor as EfficientNetFeatureExtractor
import pickle
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from evaluateChannel import evaluate_channel_selection_nmi


def compute_nmi_for_channel(feature_vector, labels, n_bins=20, strategy='quantile'):
    # 这里输入的特征向量就应该是单通道的那种[B,]
    # 先离散化特征
    discretizer = KBinsDiscretizer(
        n_bins = n_bins,
        encode = 'ordinal',
        strategy = strategy
    )

    # 横排变竖排 --》 [B,1]
    feature_2d = feature_vector.reshape(-1, 1)
    feature_discrete = discretizer.fit_transform(feature_2d).flatten().astype(int)      # 这里用int？，后面再看

    # 算nmi
    nmi = normalized_mutual_info_score(labels, feature_discrete)
    return nmi


def compute_kmeans_nmi_for_channel(feature_vector, labels, n_clusters=None, random_state=42):
    """Score a single channel using the same idea as evaluateChannel.py:
    Standardize -> KMeans discretization -> NMI(labels, cluster_id).
    """

    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)

    x = np.asarray(feature_vector).reshape(-1, 1)
    discretizer = KBinsDiscretizer(
        n_bins=n_clusters,
        encode='ordinal',
        strategy='kmeans',
        random_state=random_state,
    )
    bins = discretizer.fit_transform(x).astype(int).ravel()
    return float(normalized_mutual_info_score(labels, bins))

def select_top_k_channels_by_nmi(features, labels, k=128, n_bins=20, strategy='quantile'):
    n_channels = features.shape[1]
    nmi_scores = np.zeros(n_channels)

    print(f"正在计算 {n_channels} 个通道的 NMI ")
    for i in range(n_channels):
        if i % 100 == 0:
            print(f"已经处理 {i} / {n_channels}  通道" )

        nmi_scores[i] = compute_nmi_for_channel(
            features[:, i], labels, n_bins=n_bins, strategy=strategy
        )
    
    # 选择 NMI 最高的前k个通道
    top_k_indices = np.argsort(nmi_scores)[-k:][::-1]  # 降序排列取前k
    return top_k_indices, nmi_scores


def select_top_k_channels_by_kmeans_nmi(
    features,
    labels,
    k=128,
    n_clusters=None,
    random_state=42,
    score_n_samples: int = 5000,
):
    """Fast per-channel KMeans-NMI using KBinsDiscretizer(strategy='kmeans').

    For 1D, k-means discretization is equivalent to assigning samples to bins
    based on 1D k-means centroids; using KBinsDiscretizer lets us fit across
    all channels efficiently.
    """

    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)

    n_channels = features.shape[1]
    n_rows = features.shape[0]

    # Stratified subsample rows for faster scoring
    if score_n_samples is None or score_n_samples <= 0 or score_n_samples >= n_rows:
        row_idx = np.arange(n_rows)
    else:
        rng = np.random.default_rng(random_state)
        classes, counts = np.unique(labels, return_counts=True)
        parts = []
        for c, cnt in zip(classes, counts):
            c_idx = np.flatnonzero(labels == c)
            take = max(1, int(round(score_n_samples * (cnt / n_rows))))
            take = min(take, c_idx.size)
            parts.append(rng.choice(c_idx, size=take, replace=False))
        row_idx = np.concatenate(parts)
        rng.shuffle(row_idx)

    X_score = features[row_idx]
    y_score = labels[row_idx]

    print(
        f"正在计算 {n_channels} 个通道的 KMeans-NMI (n_bins={n_clusters}) "
        f"using {X_score.shape[0]}/{n_rows} samples"
    )

    discretizer = KBinsDiscretizer(
        n_bins=n_clusters,
        encode='ordinal',
        strategy='kmeans',
        random_state=random_state,
    )
    Xb = discretizer.fit_transform(X_score).astype(int)  # [Ns, C]

    nmi_scores = np.zeros(n_channels, dtype=np.float32)
    for i in range(n_channels):
        if i % 200 == 0:
            print(f"已经处理 {i} / {n_channels}  通道")
        nmi_scores[i] = float(normalized_mutual_info_score(y_score, Xb[:, i]))

    top_k_indices = np.argsort(nmi_scores)[-k:][::-1]
    return top_k_indices, nmi_scores


def compute_subspace_kmeans_nmi(features_subset, labels, n_clusters, random_state=42):
    """
    计算特征子空间的 KMeans-NMI。
    输入:
        features_subset: 选中通道的特征矩阵 [N, num_selected]
        labels: 真实标签 [N]
        n_clusters: 聚类数
    返回:
        nmi: 归一化互信息
    """
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_subset)
    
    # KMeans 聚类
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3
    )
    cluster_ids = kmeans.fit_predict(features_scaled)
    
    # 计算 NMI
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    return float(nmi)


def greedy_select_channels(
    features,
    labels,
    max_k=256,
    n_clusters=None,
    random_state=42,
    score_n_samples=3000,
    checkpoints=None
):
    """
    贪心法选择通道：每一步选择能使子空间 KMeans-NMI 最大的通道。
    
    输入:
        features: 完整特征矩阵 [N, 1280]
        labels: 标签 [N]
        max_k: 最多选多少个通道
        n_clusters: KMeans 聚类数，默认用类别数
        random_state: 随机种子
        score_n_samples: 用于打分的采样数（加速）
        checkpoints: 记录这些维数下的 NMI，比如 [32, 64, 128, 256]
    
    返回:
        selected_indices: 按选择顺序排列的通道索引列表
        nmi_curve: 每选一个通道后的 NMI 值列表
        checkpoint_results: 字典，记录 checkpoints 维数下的 NMI
    """
    
    # 设置默认值
    if checkpoints is None:
        checkpoints = [32, 64, 128, 256]
    
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)
    
    n_rows = features.shape[0]
    n_channels = features.shape[1]
    
    # 分层采样，加速计算
    if score_n_samples is None or score_n_samples <= 0 or score_n_samples >= n_rows:
        row_idx = np.arange(n_rows)
    else:
        rng = np.random.default_rng(random_state)
        classes = np.unique(labels)
        parts = []
        for c in classes:
            c_idx = np.flatnonzero(labels == c)
            cnt = c_idx.size
            take = max(1, int(round(score_n_samples * (cnt / n_rows))))
            take = min(take, cnt)
            chosen = rng.choice(c_idx, size=take, replace=False)
            parts.append(chosen)
        row_idx = np.concatenate(parts)
        rng.shuffle(row_idx)
    
    X_score = features[row_idx]
    y_score = labels[row_idx]
    
    print(f"贪心选择通道（单进程），最多选 {max_k} 个")
    print(f"使用 {X_score.shape[0]}/{n_rows} 样本进行打分")
    print(f"KMeans 聚类数: {n_clusters}")
    print("")
    
    # 初始化
    selected_indices = []       # 已选通道的索引
    remaining_indices = list(range(n_channels))  # 还没选的通道索引
    nmi_curve = []              # 每一步的 NMI
    checkpoint_results = {}    # 记录 checkpoints 维数的 NMI
    
    # 外层进度条：选择了多少个通道
    outer_pbar = tqdm(total=max_k, desc="通道选择进度", unit="个")
    
    # 贪心循环
    for step in range(max_k):
        best_channel = -1
        best_nmi = -1.0
        
        # 内层进度条：遍历剩余通道
        inner_pbar = tqdm(
            remaining_indices, 
            desc=f"  第{step+1}轮搜索", 
            leave=False,
            unit="ch"
        )
        
        # 遍历所有剩余通道，找到加入后 NMI 最高的
        for channel in inner_pbar:
            # 临时加入这个通道
            temp_indices = selected_indices + [channel]
            temp_features = X_score[:, temp_indices]
            
            # 计算这个子空间的 NMI
            nmi = compute_subspace_kmeans_nmi(
                temp_features, y_score, n_clusters, random_state
            )
            
            if nmi > best_nmi:
                best_nmi = nmi
                best_channel = channel
            
            # 更新内层进度条的后缀信息
            inner_pbar.set_postfix(best_nmi=f"{best_nmi:.4f}")
        
        inner_pbar.close()
        
        # 把最好的通道加入已选列表
        selected_indices.append(best_channel)
        remaining_indices.remove(best_channel)
        nmi_curve.append(best_nmi)
        
        # 更新外层进度条
        current_k = step + 1
        outer_pbar.update(1)
        outer_pbar.set_postfix(NMI=f"{best_nmi:.4f}", ch=best_channel)
        
        # 记录 checkpoint
        if current_k in checkpoints:
            checkpoint_results[current_k] = best_nmi
    
    outer_pbar.close()
    
    print("")
    print("贪心选择完成！")
    print(f"共选择了 {len(selected_indices)} 个通道")
    
    return selected_indices, nmi_curve, checkpoint_results


# ========== 多进程并行版本 ==========

# 全局变量，用于多进程共享数据
_GLOBAL_X_SCORE = None
_GLOBAL_Y_SCORE = None
_GLOBAL_N_CLUSTERS = None
_GLOBAL_RANDOM_STATE = None
_GLOBAL_SELECTED = None


def _init_worker(X_score, y_score, n_clusters, random_state):
    """初始化每个子进程的全局变量"""
    global _GLOBAL_X_SCORE, _GLOBAL_Y_SCORE, _GLOBAL_N_CLUSTERS, _GLOBAL_RANDOM_STATE
    _GLOBAL_X_SCORE = X_score
    _GLOBAL_Y_SCORE = y_score
    _GLOBAL_N_CLUSTERS = n_clusters
    _GLOBAL_RANDOM_STATE = random_state


def _eval_channel_worker(args):
    """
    工作函数：评估单个通道加入后的 NMI。
    输入: (channel_index, selected_indices_list)
    输出: (channel_index, nmi)
    """
    channel, selected_list = args
    
    # 临时加入这个通道
    temp_indices = selected_list + [channel]
    temp_features = _GLOBAL_X_SCORE[:, temp_indices]
    
    # 计算 NMI
    nmi = compute_subspace_kmeans_nmi(
        temp_features, 
        _GLOBAL_Y_SCORE, 
        _GLOBAL_N_CLUSTERS, 
        _GLOBAL_RANDOM_STATE
    )
    
    return (channel, nmi)


def greedy_select_channels_parallel(
    features,
    labels,
    max_k=256,
    n_clusters=None,
    random_state=42,
    score_n_samples=3000,
    checkpoints=None,
    n_workers=None
):
    """
    多进程并行版本的贪心通道选择。
    
    输入:
        features: 完整特征矩阵 [N, 1280]
        labels: 标签 [N]
        max_k: 最多选多少个通道
        n_clusters: KMeans 聚类数，默认用类别数
        random_state: 随机种子
        score_n_samples: 用于打分的采样数（加速）
        checkpoints: 记录这些维数下的 NMI，比如 [32, 64, 128, 256]
        n_workers: 并行进程数，默认为 CPU 核心数
    
    返回:
        selected_indices: 按选择顺序排列的通道索引列表
        nmi_curve: 每选一个通道后的 NMI 值列表
        checkpoint_results: 字典，记录 checkpoints 维数下的 NMI
    """
    import multiprocessing as mp
    
    # 设置默认值
    if checkpoints is None:
        checkpoints = [32, 64, 128, 256]
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)
    
    n_rows = features.shape[0]
    n_channels = features.shape[1]
    
    # 分层采样，加速计算
    if score_n_samples is None or score_n_samples <= 0 or score_n_samples >= n_rows:
        row_idx = np.arange(n_rows)
    else:
        rng = np.random.default_rng(random_state)
        classes = np.unique(labels)
        parts = []
        for c in classes:
            c_idx = np.flatnonzero(labels == c)
            cnt = c_idx.size
            take = max(1, int(round(score_n_samples * (cnt / n_rows))))
            take = min(take, cnt)
            chosen = rng.choice(c_idx, size=take, replace=False)
            parts.append(chosen)
        row_idx = np.concatenate(parts)
        rng.shuffle(row_idx)
    
    X_score = features[row_idx]
    y_score = labels[row_idx]
    
    print(f"贪心选择通道（多进程版本），最多选 {max_k} 个")
    print(f"使用 {X_score.shape[0]}/{n_rows} 样本进行打分")
    print(f"KMeans 聚类数: {n_clusters}")
    print(f"并行进程数: {n_workers}")
    print("")
    
    # 初始化
    selected_indices = []
    remaining_indices = list(range(n_channels))
    nmi_curve = []
    checkpoint_results = {}
    
    # 进度条：选择了多少个通道
    pbar = tqdm(total=max_k, desc="通道选择进度", unit="个")
    
    # 贪心循环
    for step in range(max_k):
        # 准备并行任务
        tasks = []
        for channel in remaining_indices:
            tasks.append((channel, list(selected_indices)))
        
        # 创建进程池并行评估
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(X_score, y_score, n_clusters, random_state)
        ) as pool:
            results = pool.map(_eval_channel_worker, tasks)
        
        # 找到最好的通道
        best_channel = -1
        best_nmi = -1.0
        for channel, nmi in results:
            if nmi > best_nmi:
                best_nmi = nmi
                best_channel = channel
        
        # 把最好的通道加入已选列表
        selected_indices.append(best_channel)
        remaining_indices.remove(best_channel)
        nmi_curve.append(best_nmi)
        
        # 更新进度条
        current_k = step + 1
        pbar.update(1)
        pbar.set_postfix(NMI=f"{best_nmi:.4f}", ch=best_channel, remaining=len(remaining_indices))
        
        # 记录 checkpoint
        if current_k in checkpoints:
            checkpoint_results[current_k] = best_nmi
    
    pbar.close()
    
    print("")
    print("贪心选择完成！")
    print(f"共选择了 {len(selected_indices)} 个通道")
    
    return selected_indices, nmi_curve, checkpoint_results


def plot_nmi_curve(nmi_curve, checkpoints, save_path="./results/nmi_curve.png"):
    """
    画 NMI 随维数变化的曲线图。
    
    输入:
        nmi_curve: 每一步的 NMI 值列表
        checkpoints: 要标记的维数列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import os
    
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 画图
    x = list(range(1, len(nmi_curve) + 1))
    y = nmi_curve
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Greedy KMeans-NMI')
    
    # 标记 checkpoints
    for cp in checkpoints:
        if cp <= len(nmi_curve):
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
            plt.scatter([cp], [nmi_curve[cp-1]], color='r', s=100, zorder=5)
            plt.annotate(
                f'k={cp}\nNMI={nmi_curve[cp-1]:.4f}',
                xy=(cp, nmi_curve[cp-1]),
                xytext=(cp+5, nmi_curve[cp-1]),
                fontsize=9
            )
    
    plt.xlabel('Number of Selected Channels', fontsize=12)
    plt.ylabel('KMeans-NMI', fontsize=12)
    plt.title('NMI vs Number of Selected Channels (Greedy Selection)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"NMI 曲线图已保存到: {save_path}")


# 特征提取+PCA降维
def extract_and_reduce_features():
    base_dir = "./dataset/ali"
    train_dir = os.path.join(base_dir,"train")

    # 使用验证转换（无数据增强）
    _, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=False,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练好的模型
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load("./results/OriginalModel.pth",map_location=device))
    model = model.to(device)
    model.eval()
    
    # 构建特征提取器：去掉classifier,保留feature + avgpool
    extractor = EfficientNetFeatureExtractor(model, pool=True, flatten=False)

    all_features = []
    all_labels = []

    print("Extracting Features...")
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)   #[B, 1280, 1, 1]
            feats = feats.view(feats.size(0),-1).cpu().numpy()  #[B, 1280]
            all_features.append(feats)
            all_labels.append(labels.numpy())
            
    X = np.concatenate(all_features, axis=0)	# [N,1280]
    np.save("./Features/features_1280.npy",X)
    y = np.concatenate(all_labels, axis=0)	#[N,]

    print(f"Feature matrix shape:{X.shape}")

    return X,y   #将特征和标签返回出去 
    
    # import scipy.stats
    
    # correlations = np.zeros(1280)
    # for i in range(1280):
    #     correlations[i], _ = scipy.stats.pearsonr(X[:,i],labels)
    
    # # 获取相关度最高的top k通道索引
    # k = 128
    # top_k_indices = np.argsort(np.abs(correlations))[-k:][::-1]

    # top_k_features = X[:,top_k_indices]

    # # PCA降维
    # from sklearn.decomposition import PCA
    # print("Performing PCA to 128 demensions....")
    # pca = PCA(n_components=128, svd_solver="full")
    # X_reduced = pca.fit_transform(X)
    # print(f"Reduced feature shape: {X_reduced.shape}")
    # print(f"Explained variance ratio (top 5): {pca.explained_variance_ratio_[:5]}")
    # print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    # 保存结果
    # save_dir = "./results"
    # np.save(os.path.join(save_dir, "features_pca128.npy"), X_reduced)
    # np.save(os.path.join(save_dir, "labels.npy"), y)

    # PCA模型
    # import joblib
    # joblib.dump(pca, os.path.join(save_dir, "pca_model.pth"))
    # print("Features and PCA model saved.")

if __name__ == "__main__":
    # 给定feature[B, 1280]  -----  EfficientNet提取的特征
    # labels：[B]
    
    # ========== 配置参数 ==========
    MAX_K = 256                 # 最多选多少个通道
    SCORE_N_SAMPLES = 3000      # 用于打分的采样数（加速）
    CHECKPOINTS = [32, 64, 128, 256]  # 记录这些维数的 NMI
    USE_GREEDY = True           # True: 贪心法, False: 逐通道打分法
    USE_PARALLEL = True         # True: 多进程并行, False: 单进程
    N_WORKERS = None            # 并行进程数, None 表示用全部 CPU 核心
    
    # ========== 提取特征 ==========
    features, labels = extract_and_reduce_features()
    
    # ========== 通道选择 ==========
    if USE_GREEDY:
        # 贪心法：每一步选择能使子空间 NMI 最大的通道
        print("=" * 50)
        if USE_PARALLEL:
            print("使用贪心法选择通道（多进程并行）")
        else:
            print("使用贪心法选择通道（单进程）")
        print("=" * 50)
        
        if USE_PARALLEL:
            # 多进程版本
            selected_indices, nmi_curve, checkpoint_results = greedy_select_channels_parallel(
                features=features,
                labels=labels,
                max_k=MAX_K,
                n_clusters=None,
                random_state=42,
                score_n_samples=SCORE_N_SAMPLES,
                checkpoints=CHECKPOINTS,
                n_workers=N_WORKERS
            )
        else:
            # 单进程版本
            selected_indices, nmi_curve, checkpoint_results = greedy_select_channels(
                features=features,
                labels=labels,
                max_k=MAX_K,
                n_clusters=None,
                random_state=42,
                score_n_samples=SCORE_N_SAMPLES,
                checkpoints=CHECKPOINTS
            )
        
        # 画 NMI 曲线图
        plot_nmi_curve(
            nmi_curve=nmi_curve,
            checkpoints=CHECKPOINTS,
            save_path="./results/nmi_curve.png"
        )
        
        # 打印 checkpoint 结果
        print("")
        print("各维数下的 NMI:")
        for k in CHECKPOINTS:
            if k in checkpoint_results:
                print(f"  k={k}: NMI={checkpoint_results[k]:.6f}")
        
        # 保存选择器
        selector_data = {
            'selected_indices': selected_indices,
            'nmi_curve': nmi_curve,
            'checkpoint_results': checkpoint_results,
            'max_k': MAX_K,
            'strategy': 'greedy_kmeans_nmi'
        }
        
    else:
        # 逐通道打分法（原来的方法）
        print("=" * 50)
        print("使用逐通道打分法选择通道")
        print("=" * 50)
        
        top_128_indices, all_nmi_scores = select_top_k_channels_by_kmeans_nmi(
            features, labels, k=128, n_clusters=None, random_state=42
        )
        
        selected_indices = list(top_128_indices)
        
        selector_data = {
            'selected_indices': selected_indices,
            'nmi_scores': all_nmi_scores,
            'n_clusters': None,
            'n_bins': int(np.unique(labels).size),
            'strategy': 'per_channel_kmeans',
            'k': 128
        }
    
    # ========== 保存选择器 ==========
    with open('nmi_channel_selector.pkl', 'wb') as f:
        pickle.dump(selector_data, f)
    print(f"选择器保存到了 'nmi_channel_selector.pkl'")
    
    # ========== 评估不同维数 ==========
    print("")
    print("=" * 50)
    print("评估筛选效果")
    print("=" * 50)
    
    for k in CHECKPOINTS:
        if k > len(selected_indices):
            continue
        
        indices_k = selected_indices[:k]
        selected_features_k = features[:, indices_k]
        
        print(f"\n--- 评估 k={k} ---")
        result = evaluate_channel_selection_nmi(
            original_features=features,
            selected_features=selected_features_k,
            labels=labels,
            n_clusters=None,
            visualize_tsne=(k == 128),  # 只在 k=128 时画 t-SNE
            tsne_out_dir="./results/t-sne",
        )

