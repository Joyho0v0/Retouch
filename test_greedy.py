#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试贪心通道选择（独立脚本，不导入 train.py）
支持多进程并行版本，带进度条
"""

import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def compute_subspace_kmeans_nmi(features_subset, labels, n_clusters, random_state=42):
    """计算子空间的 KMeans-NMI"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_subset)
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=256,
        n_init=3
    )
    cluster_ids = kmeans.fit_predict(features_scaled)
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    return float(nmi)


# 全局变量，用于多进程共享数据
_GLOBAL_X = None
_GLOBAL_Y = None
_GLOBAL_N_CLUSTERS = None
_GLOBAL_RANDOM_STATE = None


def _init_worker(X, y, n_clusters, random_state):
    """初始化子进程的全局变量"""
    global _GLOBAL_X, _GLOBAL_Y, _GLOBAL_N_CLUSTERS, _GLOBAL_RANDOM_STATE
    _GLOBAL_X = X
    _GLOBAL_Y = y
    _GLOBAL_N_CLUSTERS = n_clusters
    _GLOBAL_RANDOM_STATE = random_state


def _eval_channel(args):
    """评估单个通道"""
    channel, selected_list = args
    temp_indices = selected_list + [channel]
    temp_features = _GLOBAL_X[:, temp_indices]
    nmi = compute_subspace_kmeans_nmi(
        temp_features, _GLOBAL_Y, _GLOBAL_N_CLUSTERS, _GLOBAL_RANDOM_STATE
    )
    return (channel, nmi)


def greedy_select_parallel(X, y, max_k=10, n_clusters=None, random_state=42, n_workers=None):
    """多进程并行版贪心选择"""
    
    if n_clusters is None:
        n_clusters = int(np.unique(y).size)
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    n_channels = X.shape[1]
    selected = []
    remaining = list(range(n_channels))
    curve = []
    
    print(f"开始贪心选择（多进程），最多选 {max_k} 个通道")
    print(f"样本数: {X.shape[0]}, 通道数: {n_channels}, 聚类数: {n_clusters}")
    print(f"并行进程数: {n_workers}")
    print()
    
    # 进度条
    pbar = tqdm(total=max_k, desc="通道选择进度", unit="个")
    
    for step in range(max_k):
        # 准备任务
        tasks = []
        for ch in remaining:
            tasks.append((ch, list(selected)))
        
        # 并行评估
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(X, y, n_clusters, random_state)
        ) as pool:
            results = pool.map(_eval_channel, tasks)
        
        # 找最好的
        best_ch = -1
        best_nmi = -1.0
        for ch, nmi in results:
            if nmi > best_nmi:
                best_nmi = nmi
                best_ch = ch
        
        selected.append(best_ch)
        remaining.remove(best_ch)
        curve.append(best_nmi)
        
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix(NMI=f"{best_nmi:.4f}", ch=best_ch)
    
    pbar.close()
    print()
    print("贪心选择完成！")
    return selected, curve


if __name__ == "__main__":
    print("加载数据...")
    X = np.load('results/features_1280.npy')
    y = np.load('results/labels.npy')
    
    print(f"原始数据: X.shape={X.shape}, y.shape={y.shape}")
    
    # 采样加速
    n_samples = 1000
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_s = X[idx]
    y_s = y[idx]
    
    print(f"采样后: X_s.shape={X_s.shape}")
    print()
    
    # 测试多进程并行版本，选 20 个通道
    selected, curve = greedy_select_parallel(
        X_s, y_s, 
        max_k=20, 
        n_clusters=None, 
        random_state=42,
        n_workers=None  # 用全部 CPU 核心
    )
    
    print()
    print("选中的通道:", selected[:10], "...")
    print("NMI 曲线:", [round(v, 4) for v in curve])
    
    # 对比原始 1280 维的 NMI
    orig_nmi = compute_subspace_kmeans_nmi(X_s, y_s, n_clusters=int(np.unique(y_s).size), random_state=42)
    print(f"\n原始 1280 维 NMI: {orig_nmi:.6f}")
    print(f"选 20 维后 NMI: {curve[-1]:.6f}")
    print(f"提升: {curve[-1] - orig_nmi:+.6f}")
