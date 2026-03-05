"""
consumeSelect.py  —— 基于已有 pkl 继续贪心通道选择

用法：
    当 ChannelSelect.py 已经跑完 MAX_K=64（或其他较小值）并保存了
    nmi_channel_selector.pkl 后，可以用本脚本继续往后选更多通道，
    而不需要从头重跑前面已经选好的部分。

原理：
    1. 加载 pkl，取出 selected_indices（已选通道）和 nmi_curve（验证集 NMI 曲线）
    2. 从 step = len(selected_indices)+1 开始，继续贪心选择，直到 NEW_MAX_K
    3. 同步计算 Ali 测试集 / Megvii 测试集 NMI 曲线
    4. 画图、评估、保存更新后的 pkl

注意：
    - 为了保证续选与原选择一致，random_state 和 score_n_samples 必须与
      ChannelSelect.py 保持一致（默认都是 42 / 3000）。
    - 采样逻辑只取决于 (features, labels, random_state, score_n_samples)，
      在相同验证集上用相同种子，采样结果完全一致。
"""

from ChannelSelect import (
    extract_val_features,
    extract_test_features,
    extract_megvii_test_features,
    compute_subspace_kmeans_nmi,
    plot_nmi_curve,
    _init_worker,
    _eval_channel_worker,
)
from evaluateChannel import evaluate_channel_selection_nmi
from train import *          # np, torch, tqdm, DataLoader, ...
import pickle
import multiprocessing as mp


# ========== 配置参数 ==========
PKL_LOAD_PATH = "./nmi_channel_selector.pkl"   # 已有的 pkl（由 ChannelSelect.py 生成）
PKL_SAVE_PATH = "./nmi_channel_selector.pkl"   # 更新后保存的路径（默认覆盖原文件）

NEW_MAX_K = 128              # 希望最终选到多少个通道（例如原来 64，现在想扩展到 128）
SCORE_N_SAMPLES = 3000       # 与 ChannelSelect.py 保持一致
RANDOM_STATE = 42            # 与 ChannelSelect.py 保持一致
N_WORKERS = 20               # 并行进程数
CHECKPOINTS = [32, 64, 128, 256]  # 记录这些维数的 NMI


def greedy_resume_parallel(
    features,
    labels,
    initial_selected,
    initial_nmi_curve,
    new_max_k,
    n_clusters=None,
    random_state=42,
    score_n_samples=3000,
    checkpoints=None,
    n_workers=None,
):
    """
    从已有的贪心结果继续选择通道。

    输入:
        features:          完整特征矩阵 [N, 1280]
        labels:            标签 [N]
        initial_selected:  已选通道索引列表（按贪心顺序）
        initial_nmi_curve: 已选通道对应的验证集 NMI 曲线
        new_max_k:         希望最终选到多少个通道
        其余参数同 greedy_select_channels_parallel

    返回:
        selected_indices:    完整的通道索引列表（包含之前 + 新选的）
        nmi_curve:           完整的 NMI 曲线（包含之前 + 新选的）
        checkpoint_results:  字典，记录 checkpoints 维数下的 NMI
    """
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

    # ---- 分层采样（与 ChannelSelect.py 完全一致的逻辑） ----
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

    # ---- 从已有结果初始化 ----
    already_done = len(initial_selected)
    selected_indices = list(initial_selected)
    nmi_curve = list(initial_nmi_curve)

    remaining_indices = list(range(n_channels))
    for ch in selected_indices:
        remaining_indices.remove(ch)

    # 填充已有的 checkpoint
    checkpoint_results = {}
    for cp in checkpoints:
        if cp <= already_done:
            checkpoint_results[cp] = nmi_curve[cp - 1]

    steps_to_do = new_max_k - already_done
    if steps_to_do <= 0:
        print(f"已选通道数 ({already_done}) >= 目标 ({new_max_k})，无需继续。")
        return selected_indices, nmi_curve, checkpoint_results

    print(f"续选通道：从第 {already_done + 1} 个选到第 {new_max_k} 个")
    print(f"使用 {X_score.shape[0]}/{n_rows} 样本进行打分")
    print(f"KMeans 聚类数: {n_clusters}")
    print(f"并行进程数: {n_workers}")
    print(f"剩余候选通道: {len(remaining_indices)}")
    print("")

    pbar = tqdm(
        total=steps_to_do,
        desc="续选通道进度",
        unit="个",
        initial=0,
    )

    for step_i in range(steps_to_do):
        global_step = already_done + step_i  # 0-based 全局步数

        # 准备并行任务
        tasks = []
        for channel in remaining_indices:
            tasks.append((channel, list(selected_indices)))

        # 多进程评估
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(X_score, y_score, n_clusters, random_state),
        ) as pool:
            results = pool.map(_eval_channel_worker, tasks)

        # 找最好的通道
        best_channel = -1
        best_nmi = -1.0
        for channel, nmi in results:
            if nmi > best_nmi:
                best_nmi = nmi
                best_channel = channel

        selected_indices.append(best_channel)
        remaining_indices.remove(best_channel)
        nmi_curve.append(best_nmi)

        current_k = global_step + 1  # 1-based
        pbar.update(1)
        pbar.set_postfix(
            k=current_k, NMI=f"{best_nmi:.4f}", ch=best_channel,
            remaining=len(remaining_indices),
        )

        if current_k in checkpoints:
            checkpoint_results[current_k] = best_nmi

    pbar.close()

    print("")
    print("续选完成！")
    print(f"共选择了 {len(selected_indices)} 个通道（之前 {already_done} + 新增 {steps_to_do}）")

    return selected_indices, nmi_curve, checkpoint_results


if __name__ == "__main__":
    # ========== 加载已有 pkl ==========
    print("=" * 60)
    print("consumeSelect  —— 基于已有 pkl 继续贪心通道选择")
    print("=" * 60)
    print("")

    with open(PKL_LOAD_PATH, "rb") as f:
        old_data = pickle.load(f)

    old_selected = old_data["selected_indices"]
    old_val_nmi_curve = old_data["nmi_curve"]
    old_max_k = old_data.get("max_k", len(old_selected))

    print(f"已加载 pkl: {PKL_LOAD_PATH}")
    print(f"  策略: {old_data.get('strategy', 'unknown')}")
    print(f"  已选通道数: {len(old_selected)}")
    print(f"  原始 max_k: {old_max_k}")
    print(f"  目标 new_max_k: {NEW_MAX_K}")
    print("")

    if len(old_selected) >= NEW_MAX_K:
        print(f"[跳过] 已选通道数 ({len(old_selected)}) >= 目标 ({NEW_MAX_K})，无需继续。")
        print("如需选更多通道，请增大 NEW_MAX_K。")
        exit(0)

    # ========== 提取特征 ==========
    val_features, val_labels = extract_val_features()
    test_features, test_labels = extract_test_features()
    meg_features, meg_labels = extract_megvii_test_features()

    # ========== 续选通道（基于验证集） ==========
    print("")
    print("=" * 50)
    print("续选贪心通道（多进程并行）")
    print("=" * 50)

    selected_indices, val_nmi_curve, val_checkpoint_results = greedy_resume_parallel(
        features=val_features,
        labels=val_labels,
        initial_selected=old_selected,
        initial_nmi_curve=old_val_nmi_curve,
        new_max_k=NEW_MAX_K,
        n_clusters=None,
        random_state=RANDOM_STATE,
        score_n_samples=SCORE_N_SAMPLES,
        checkpoints=CHECKPOINTS,
        n_workers=N_WORKERS,
    )

    # ========== 在 Ali 测试集上计算完整 NMI 曲线 ==========
    # 因为之前 pkl 里可能有部分 test/megvii 曲线，但通道顺序可能已经扩展了，
    # 这里统一重新算完整曲线（对前面已有的部分也重新算一遍，保证一致性）。
    print("")
    print("在 Ali 测试集上计算完整 NMI 曲线...")
    n_clusters_test = int(np.unique(test_labels).size)
    test_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="Ali Test NMI Curve"):
        subset = test_features[:, selected_indices[:step]]
        nmi_val = compute_subspace_kmeans_nmi(subset, test_labels, n_clusters_test)
        test_nmi_curve.append(nmi_val)

    test_checkpoint_results = {}
    for cp in CHECKPOINTS:
        if cp <= len(test_nmi_curve):
            test_checkpoint_results[cp] = test_nmi_curve[cp - 1]

    # ========== 在 Megvii 测试集上计算完整 NMI 曲线 ==========
    print("")
    print("在 Megvii 测试集上计算完整 NMI 曲线（跨域评估）...")
    n_clusters_meg = int(np.unique(meg_labels).size)
    megvii_test_nmi_curve = []
    for step in tqdm(range(1, len(selected_indices) + 1), desc="Megvii Test NMI Curve"):
        subset = meg_features[:, selected_indices[:step]]
        nmi_val = compute_subspace_kmeans_nmi(subset, meg_labels, n_clusters_meg)
        megvii_test_nmi_curve.append(nmi_val)

    megvii_checkpoint_results = {}
    for cp in CHECKPOINTS:
        if cp <= len(megvii_test_nmi_curve):
            megvii_checkpoint_results[cp] = megvii_test_nmi_curve[cp - 1]

    # ========== 计算 OriginalModel 全 1280 通道 NMI ==========
    print("")
    print("计算 OriginalModel 全 1280 通道 NMI...")
    n_clusters_val = int(np.unique(val_labels).size)
    n_clusters_test = int(np.unique(test_labels).size)
    n_clusters_meg = int(np.unique(meg_labels).size)
    original_nmi_val = compute_subspace_kmeans_nmi(val_features, val_labels, n_clusters_val)
    original_nmi_test = compute_subspace_kmeans_nmi(test_features, test_labels, n_clusters_test)
    original_nmi_megvii = compute_subspace_kmeans_nmi(meg_features, meg_labels, n_clusters_meg)
    print(f"  OriginalModel (1280ch) Val={original_nmi_val:.4f}, "
          f"Test={original_nmi_test:.4f}, Megvii={original_nmi_megvii:.4f}")

    # ========== 画 NMI 曲线图 ==========
    plot_nmi_curve(
        nmi_curve=val_nmi_curve,
        checkpoints=CHECKPOINTS,
        save_path="./results/nmi_curve.png",
        test_nmi_curve=test_nmi_curve,
        megvii_test_nmi_curve=megvii_test_nmi_curve,
        original_nmi_val=original_nmi_val,
        original_nmi_test=original_nmi_test,
        original_nmi_megvii=original_nmi_megvii,
    )

    # ========== 打印 checkpoint 结果 ==========
    print("")
    print("各维数下的 NMI（验证集）:")
    for k in CHECKPOINTS:
        if k in val_checkpoint_results:
            print(f"  k={k}: NMI={val_checkpoint_results[k]:.6f}")

    print("")
    print("各维数下的 NMI（Ali 测试集）:")
    for k in CHECKPOINTS:
        if k in test_checkpoint_results:
            print(f"  k={k}: NMI={test_checkpoint_results[k]:.6f}")

    print("")
    print("各维数下的 NMI（Megvii 测试集，跨域）:")
    for k in CHECKPOINTS:
        if k in megvii_checkpoint_results:
            print(f"  k={k}: NMI={megvii_checkpoint_results[k]:.6f}")

    # ========== 保存更新后的 pkl ==========
    selector_data = {
        "selected_indices": selected_indices,
        "nmi_curve": val_nmi_curve,
        "nmi_curve_test": test_nmi_curve,
        "nmi_curve_megvii_test": megvii_test_nmi_curve,
        "checkpoint_results": val_checkpoint_results,
        "checkpoint_results_test": test_checkpoint_results,
        "checkpoint_results_megvii_test": megvii_checkpoint_results,
        "max_k": NEW_MAX_K,
        "strategy": "greedy_kmeans_nmi",
    }

    with open(PKL_SAVE_PATH, "wb") as f:
        pickle.dump(selector_data, f)
    print(f"\n选择器已保存到 '{PKL_SAVE_PATH}'")

    # ========== 评估不同维数（验证集） ==========
    print("")
    print("=" * 50)
    print("评估筛选效果（验证集）")
    print("=" * 50)

    for k in CHECKPOINTS:
        if k > len(selected_indices):
            continue

        indices_k = selected_indices[:k]
        selected_val_features_k = val_features[:, indices_k]

        print(f"\n--- 评估 k={k}（验证集） ---")
        evaluate_channel_selection_nmi(
            original_features=val_features,
            selected_features=selected_val_features_k,
            labels=val_labels,
            n_clusters=None,
            visualize_tsne=(k in CHECKPOINTS),
            tsne_out_dir="./results/t-sne",
        )
