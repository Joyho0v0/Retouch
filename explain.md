# ChannelSelect.py 代码逐行解析

## 一、关于贪心法是否会漏选"协同通道组合"的分析

### 你的疑问

> 前面已经验证过的通道索引 1,5,7 的组合以及单独的 10 号通道对模型的重要程度并不高，所以贪心第三轮选取结束后就会把 1,5,7 这三个组合给抛弃，但是实际上 1,5,7,10 进行组合反而 NMI 会大幅上升，是否会遇到无法选到 1,5,7,10 的可能性？

### 分析结论：你的担心 **部分成立**，但具体机制与你描述的不完全一样

首先需要澄清一个重要误解：**贪心法并不会"把通道 1,5,7 的组合给抛弃"**。贪心法在每一轮中不是评估固定组合，而是把每个剩余通道逐一尝试加入当前已选集合。只要通道 1,5,7,10 没有被选中，它们就一直留在候选池里，每一轮都会被重新评估。

但是，贪心法确实存在一个**根本性的局限**——它每次只添加 **一个** 通道。具体来说：

**贪心法的工作流程（以你的例子为例）：**

- **第 1 轮**：评估每个单通道 {c} 的 NMI，选出最好的。假设通道 42 最好 → 选中 {42}
- **第 2 轮**：评估 {42, c} 对所有剩余 c 的 NMI，选出最好的。假设通道 200 → 选中 {42, 200}
- **第 3 轮**：评估 {42, 200, c} 对所有剩余 c，选出最好的。假设通道 500 → 选中 {42, 200, 500}
- ...以此类推

在这个过程中，通道 1,5,7,10 始终在候选池里。每一轮都会测试"把通道 1 加入当前集合"的效果、"把通道 5 加入当前集合"的效果等等。

**问题出在哪里？**

如果通道 1,5,7,10 的协同效应要求它们**必须同时出现**才能生效（即任意子集 {1}、{1,5}、{1,5,7} 都不比其他候选通道好），那么：

1. 第 1 轮：通道 1 单独的 NMI 不高 → 不被选中（正确，其他通道确实更好）
2. 后续轮次：把通道 1 加入已有集合 {42, 200, ...} 中，NMI 提升也不明显 → 仍不被选中
3. 通道 5,7,10 同理
4. 因为贪心法一次只加一个，它永远无法"同时加入 1,5,7,10"来发现这个协同效应

**这种情况在实际中常见吗？**

对于 EfficientNet 提取的 1280 维特征而言，**纯粹的多通道协同效应（没有任何递增信号）是很少见的**。更常见的情况是：

- 通道 1 单独 NMI 一般，但加入一个好的集合后能提供额外信息 → 贪心法能捕捉到
- 通道 1 和通道 5 有一定的互补性，即 {已选集合, 1} 比 {已选集合, 其他} 好 → 贪心法也能捕捉到

只有那种"四个通道缺一不可、任意子集完全没有信号"的极端情况，贪心法才会完全失败。这在高维连续特征空间中概率很低。

**总结：**

| 场景 | 贪心法能否发现？ |
|------|-----------------|
| 通道 A 单独就有高 NMI | ✅ 第 1 轮就能选中 |
| 通道 A 单独一般，但和已选集合互补 | ✅ 在某一轮中能选中 |
| 通道 A,B 必须同时出现才有效（成对协同） | ❌ 无法发现 |
| 通道 A,B,C,D 必须全部出现才有效（多路协同） | ❌ 无法发现 |

所以你的担心是有道理的，但概率不高。为了弥补这个缺陷，随机搜索法（RandomSelect.py）是一个合理的补充方案——它每次直接生成完整的 128 通道组合，不受"逐个添加"的约束，理论上能发现任意协同组合（只要搜索次数足够多）。

---

## 二、ChannelSelect.py 代码逐行解析

### 导入部分（第 1~8 行）

```python
from train import *
```
导入 train.py 中的所有内容。train.py 中定义了 `build_model`、`get_transforms`、`DataLoader`、`datasets`、`torch`、`np`、`os`、`tqdm` 等常用工具和模型构建函数。通过 `*` 导入，后面可以直接使用这些名字。

```python
from EfficientNet_B0 import FeatureExtractor as EfficientNetFeatureExtractor
```
从 EfficientNet_B0.py 中导入 `FeatureExtractor` 类，并重命名为 `EfficientNetFeatureExtractor`。这个类用于从 EfficientNet 模型中提取中间特征（去掉分类头，只保留特征提取部分和全局平均池化层）。

```python
import pickle
```
导入 pickle 模块，用于将通道选择结果序列化保存到 `.pkl` 文件。

```python
from sklearn.metrics import normalized_mutual_info_score
```
导入归一化互信息（NMI）计算函数。NMI 用于衡量聚类结果与真实标签之间的一致性，值域 [0, 1]，越大表示聚类越接近真实分类。

```python
from sklearn.preprocessing import KBinsDiscretizer
```
导入分箱离散化器。它可以将连续特征值分成若干个离散的区间（箱子），支持多种分箱策略（均匀分箱、分位数分箱、KMeans 分箱等）。

```python
from sklearn.cluster import MiniBatchKMeans
```
导入小批量 KMeans 聚类算法。相比标准 KMeans，MiniBatchKMeans 每次只用一部分样本更新聚类中心，速度更快，适用于大数据场景。

```python
from sklearn.preprocessing import StandardScaler
```
导入标准化器，用于将特征缩放到均值为 0、标准差为 1。在做 KMeans 聚类前先标准化，可以避免不同通道的数值范围差异影响聚类结果。

```python
from evaluateChannel import evaluate_channel_selection_nmi
```
导入通道选择评估函数，用于在最后评估选出的通道子集的效果（包含 t-SNE 可视化功能）。

---

### 函数 `compute_nmi_for_channel`（第 11~27 行）

**功能：** 计算单个通道的特征向量与真实标签之间的 NMI。先将连续特征离散化为若干区间，再计算离散化结果与标签的 NMI。

```python
def compute_nmi_for_channel(feature_vector, labels, n_bins=20, strategy='quantile'):
```
定义函数，接收参数：
- `feature_vector`：一个通道的特征向量，形状 [B,]（B 是样本数）
- `labels`：真实标签，形状 [B,]
- `n_bins`：离散化的区间数，默认 20
- `strategy`：分箱策略，默认 `'quantile'`（分位数分箱，每个箱中的样本数大致相等）

```python
    discretizer = KBinsDiscretizer(
        n_bins = n_bins,
        encode = 'ordinal',
        strategy = strategy
    )
```
创建分箱离散化器：
- `n_bins=20`：分成 20 个区间
- `encode='ordinal'`：输出为区间编号（0, 1, 2, ...），而不是独热编码
- `strategy='quantile'`：使用分位数策略，使每个区间包含大致相同数量的样本

```python
    feature_2d = feature_vector.reshape(-1, 1)
```
将一维特征向量 [B,] 转为二维 [B, 1]，因为 `KBinsDiscretizer` 要求输入至少是二维数组。

```python
    feature_discrete = discretizer.fit_transform(feature_2d).flatten().astype(int)
```
先用 `fit_transform` 对特征进行拟合并转换（即学习分箱边界并应用），然后 `flatten()` 把 [B, 1] 压回 [B,]，最后转为整数类型。

```python
    nmi = normalized_mutual_info_score(labels, feature_discrete)
    return nmi
```
计算离散化后的特征值（相当于"伪聚类标签"）与真实标签之间的 NMI，然后返回结果。

---

### 函数 `compute_kmeans_nmi_for_channel`（第 30~52 行）

**功能：** 与上一个函数类似，但使用 KMeans 策略进行分箱。这与 `evaluateChannel.py` 的评估方式一致。

```python
def compute_kmeans_nmi_for_channel(feature_vector, labels, n_clusters=None, random_state=42):
```
参数：
- `n_clusters`：KMeans 聚类/分箱的数目。如果不指定，自动用标签的类别数（对于二分类任务就是 2）
- `random_state`：随机种子，保证结果可复现

```python
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)
```
把标签转为 numpy 数组。如果没指定聚类数，就取标签中不同值的数量（如二分类中 n_clusters=2）。

```python
    x = np.asarray(feature_vector).reshape(-1, 1)
```
同样将特征向量转为 [B, 1] 的形状。

```python
    discretizer = KBinsDiscretizer(
        n_bins=n_clusters,
        encode='ordinal',
        strategy='kmeans',
        random_state=random_state,
    )
```
创建分箱器，这次用 `strategy='kmeans'`：它会对每个特征列（这里只有一列）做一维 KMeans 聚类，用聚类边界作为分箱边界。分箱数等于聚类数。

```python
    bins = discretizer.fit_transform(x).astype(int).ravel()
    return float(normalized_mutual_info_score(labels, bins))
```
拟合并转换，得到每个样本属于哪个箱子，然后计算 NMI。

---

### 函数 `select_top_k_channels_by_nmi`（第 54~69 行）

**功能：** 逐通道打分法——对每个通道独立计算 NMI，然后选出 NMI 最高的前 k 个通道。

```python
def select_top_k_channels_by_nmi(features, labels, k=128, n_bins=20, strategy='quantile'):
    n_channels = features.shape[1]
    nmi_scores = np.zeros(n_channels)
```
获取通道总数（1280），创建一个全零数组来存放每个通道的 NMI 分数。

```python
    for i in range(n_channels):
        if i % 100 == 0:
            print(f"已经处理 {i} / {n_channels}  通道" )
        nmi_scores[i] = compute_nmi_for_channel(
            features[:, i], labels, n_bins=n_bins, strategy=strategy
        )
```
遍历所有 1280 个通道，每个通道取出其特征列 `features[:, i]`（形状 [N,]），调用 `compute_nmi_for_channel` 计算该通道的 NMI 得分。每处理 100 个通道打印一次进度。

```python
    top_k_indices = np.argsort(nmi_scores)[-k:][::-1]
    return top_k_indices, nmi_scores
```
`np.argsort` 返回从小到大排序的索引，取最后 k 个（最大的 k 个），再 `[::-1]` 翻转为从大到小。返回选中的通道索引和所有通道的 NMI 分数。

---

### 函数 `select_top_k_channels_by_kmeans_nmi`（第 72~130 行）

**功能：** 与上面类似的逐通道打分法，但使用 KMeans 分箱策略，并且支持对样本进行分层采样以加速计算。

```python
def select_top_k_channels_by_kmeans_nmi(
    features, labels, k=128, n_clusters=None, random_state=42, score_n_samples=5000):
```
参数新增了 `score_n_samples`：用于评分的样本数量。如果数据量太大，可以只用一部分样本来计算 NMI，加快速度。

```python
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)
    n_channels = features.shape[1]
    n_rows = features.shape[0]
```
处理默认聚类数，获取通道数和样本数。

```python
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
```
**分层采样逻辑：** 如果不需要采样（样本数不够多或没设置），就用全部样本。否则：
1. 对每个类别，按照其在数据中的比例，分配对应数量的采样名额
2. 从每个类别中随机抽取对应数量的样本索引
3. 合并所有类别的采样索引并打乱顺序

这样可以保证采样后各类别比例与原始数据一致（分层采样），避免类别不平衡导致评估偏差。

```python
    X_score = features[row_idx]
    y_score = labels[row_idx]
```
取出采样后的特征和标签。

```python
    discretizer = KBinsDiscretizer(
        n_bins=n_clusters, encode='ordinal', strategy='kmeans', random_state=random_state,
    )
    Xb = discretizer.fit_transform(X_score).astype(int)  # [Ns, C]
```
关键操作：对**所有通道同时**进行 KMeans 分箱。`KBinsDiscretizer` 会对每一列（每个通道）独立做一维 KMeans。输出 `Xb` 的形状与 `X_score` 相同，但值变成了离散的箱子编号。

```python
    nmi_scores = np.zeros(n_channels, dtype=np.float32)
    for i in range(n_channels):
        if i % 200 == 0:
            print(f"已经处理 {i} / {n_channels}  通道")
        nmi_scores[i] = float(normalized_mutual_info_score(y_score, Xb[:, i]))
```
遍历每个通道，计算该通道离散化后的值与标签的 NMI。

```python
    top_k_indices = np.argsort(nmi_scores)[-k:][::-1]
    return top_k_indices, nmi_scores
```
同样选出 NMI 最高的前 k 个通道。

---

### 函数 `compute_subspace_kmeans_nmi`（第 133~157 行）

**功能：** 计算一组选中通道构成的子空间的整体 NMI。这是贪心法和评估中最核心的打分函数。

```python
def compute_subspace_kmeans_nmi(features_subset, labels, n_clusters, random_state=42):
```
参数：
- `features_subset`：选中通道的特征矩阵 [N, num_selected]
- `labels`：真实标签 [N,]
- `n_clusters`：KMeans 聚类数

```python
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_subset)
```
先做标准化：每个通道减去均值、除以标准差。这样所有通道的数值范围一致，KMeans 聚类时不会被某个数值范围特别大的通道主导。

```python
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3
    )
    cluster_ids = kmeans.fit_predict(features_scaled)
```
创建小批量 KMeans 聚类器：
- `n_clusters`：聚类数（通常等于类别数 2）
- `batch_size=1024`：每次迭代用 1024 个样本更新聚类中心
- `n_init=3`：用 3 组不同的随机初始中心重复聚类，取最好的结果

`fit_predict` 同时拟合模型并返回每个样本的聚类标签。

```python
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    return float(nmi)
```
计算聚类标签与真实标签的 NMI。这里是在**多维子空间**上做 KMeans，与前面逐通道的方式不同——前面是对单个通道做一维分箱，这里是对多个通道联合做多维聚类。

---

### 函数 `greedy_select_channels`（第 160~284 行）

**功能：** 贪心法选择通道的核心算法（单进程版本）。

```python
def greedy_select_channels(
    features, labels, max_k=128, n_clusters=None, random_state=42,
    score_n_samples=3000, checkpoints=None):
```
参数含义：
- `features [N, 1280]`：完整的 1280 维特征矩阵
- `labels [N,]`：标签
- `max_k`：最多选择多少个通道
- `score_n_samples`：采样数（为了加速，不用全部数据评分）
- `checkpoints`：指定在哪些通道数时记录 NMI

```python
    if checkpoints is None:
        checkpoints = [32, 64, 128]
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = int(n_clusters)
    n_rows = features.shape[0]
    n_channels = features.shape[1]
```
设置默认值，处理聚类数。

**分层采样部分（同前面的逻辑，此处省略重复解释）**

```python
    selected_indices = []
    remaining_indices = list(range(n_channels))
    nmi_curve = []
    checkpoint_results = {}
```
初始化四个关键变量：
- `selected_indices`：已选中的通道索引列表（按选择顺序）
- `remaining_indices`：还没被选中的通道索引列表（初始为 0~1279）
- `nmi_curve`：记录每一步选择后的最优 NMI
- `checkpoint_results`：在指定通道数时的 NMI 记录

```python
    outer_pbar = tqdm(total=max_k, desc="通道选择进度", unit="个")
```
创建外层进度条，显示总共要选多少个通道。

**贪心搜索的核心循环：**

```python
    for step in range(max_k):       # 外层：共选 max_k 个通道
        best_channel = -1
        best_nmi = -1.0
```
每一轮开始时，重置"本轮最佳通道"和"本轮最佳 NMI"。

```python
        inner_pbar = tqdm(remaining_indices, desc=f"  第{step+1}轮搜索", leave=False, unit="ch")
```
创建内层进度条，遍历所有剩余通道。

```python
        for channel in inner_pbar:            # 内层：遍历每个剩余通道
            temp_indices = selected_indices + [channel]  # 临时把这个通道加入已选集合
            temp_features = X_score[:, temp_indices]     # 取出对应的特征列
```
**这是贪心法的关键操作：** 对于每个还没选中的通道，临时把它加到已选列表的末尾，形成一个新的候选子集。

```python
            nmi = compute_subspace_kmeans_nmi(
                temp_features, y_score, n_clusters, random_state
            )
```
在这个候选子集上做 KMeans 聚类，计算整体 NMI。

```python
            if nmi > best_nmi:
                best_nmi = nmi
                best_channel = channel
            inner_pbar.set_postfix(best_nmi=f"{best_nmi:.4f}")
```
如果这个通道加入后的 NMI 比目前找到的最佳值还高，就更新最佳通道和最佳 NMI。

```python
        inner_pbar.close()
        selected_indices.append(best_channel)      # 正式把最好的通道加入已选列表
        remaining_indices.remove(best_channel)      # 从候选池中移除
        nmi_curve.append(best_nmi)                  # 记录本轮 NMI
```
内层循环结束后，把本轮评估出的最佳通道正式"选中"。

```python
        current_k = step + 1
        outer_pbar.update(1)
        outer_pbar.set_postfix(NMI=f"{best_nmi:.4f}", ch=best_channel)
        if current_k in checkpoints:
            checkpoint_results[current_k] = best_nmi
```
更新外层进度条，如果当前步数在 checkpoint 列表中，记录该步的 NMI。

```python
    outer_pbar.close()
    return selected_indices, nmi_curve, checkpoint_results
```
全部选择完成后，返回：选中的通道索引列表、每一步的 NMI 曲线、checkpoint 处的 NMI。

**注意：** 贪心法的时间复杂度为 O(max_k × remaining_channels × KMeans_cost)。以 max_k=128、1280 个通道为例，第一轮需要评估 1280 个候选，第二轮 1279 个...因此总评估次数约为 128 × 1216 ≈ 155,648 次 KMeans 聚类，非常耗时。

---

### 多进程并行版本（第 287~452 行）

**功能：** `greedy_select_channels_parallel` 是上面贪心法的多进程并行加速版本。

#### 全局变量和工作函数

```python
_GLOBAL_X_SCORE = None
_GLOBAL_Y_SCORE = None
_GLOBAL_N_CLUSTERS = None
_GLOBAL_RANDOM_STATE = None
```
声明全局变量，用于在多进程间共享特征数据和参数。在 Python 的 `multiprocessing.Pool` 中，每个子进程有自己的内存空间，通过 `initializer` 函数将数据设置到子进程的全局变量中，避免每次传参都要序列化大量数据。

```python
def _init_worker(X_score, y_score, n_clusters, random_state):
    global _GLOBAL_X_SCORE, _GLOBAL_Y_SCORE, _GLOBAL_N_CLUSTERS, _GLOBAL_RANDOM_STATE
    _GLOBAL_X_SCORE = X_score
    _GLOBAL_Y_SCORE = y_score
    _GLOBAL_N_CLUSTERS = n_clusters
    _GLOBAL_RANDOM_STATE = random_state
```
子进程的初始化函数。当进程池创建时，每个子进程都会调用这个函数，把共享数据存到自己的全局变量中。

```python
def _eval_channel_worker(args):
    channel, selected_list = args
    temp_indices = selected_list + [channel]
    temp_features = _GLOBAL_X_SCORE[:, temp_indices]
    nmi = compute_subspace_kmeans_nmi(
        temp_features, _GLOBAL_Y_SCORE, _GLOBAL_N_CLUSTERS, _GLOBAL_RANDOM_STATE
    )
    return (channel, nmi)
```
工作函数：接收 `(通道索引, 已选通道列表)`，评估把该通道加入后的 NMI，返回 `(通道索引, NMI)`。这个函数在子进程中执行。

#### `greedy_select_channels_parallel` 主函数

逻辑与单进程版本完全一致，区别在于内层遍历剩余通道的部分改为并行：

```python
        tasks = []
        for channel in remaining_indices:
            tasks.append((channel, list(selected_indices)))
```
构建任务列表，每个任务是 `(候选通道, 当前已选列表的拷贝)`。

```python
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(X_score, y_score, n_clusters, random_state)
        ) as pool:
            results = pool.map(_eval_channel_worker, tasks)
```
创建进程池，用 `pool.map` 并行评估所有候选通道。这样原本串行的 1280 次 KMeans 评估变成了并行执行，速度提升 n_workers 倍。

```python
        best_channel = -1
        best_nmi = -1.0
        for channel, nmi in results:
            if nmi > best_nmi:
                best_nmi = nmi
                best_channel = channel
```
从并行结果中找出最佳通道。后续逻辑与单进程版本相同。

---

### 函数 `plot_nmi_curve`（第 455~531 行）

**功能：** 画 NMI 随通道数变化的折线图，支持三条曲线。

```python
def plot_nmi_curve(nmi_curve, checkpoints, save_path="./results/nmi_curve.png",
                   test_nmi_curve=None, megvii_test_nmi_curve=None):
```
参数：
- `nmi_curve`：Ali 验证集的 NMI 曲线（天蓝色）
- `test_nmi_curve`：Ali 测试集的 NMI 曲线（橙黄色，可选）
- `megvii_test_nmi_curve`：Megvii 测试集的 NMI 曲线（番茄红，可选）

```python
    x = list(range(1, len(nmi_curve) + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(x, nmi_curve, color='skyblue', linestyle='-', linewidth=2, label='Ali Val NMI')
```
横轴为通道数（1 到 max_k），画第一条曲线（Ali 验证集）。

后面依次画 Ali 测试集和 Megvii 测试集的曲线（如果有数据的话），然后在 checkpoint 位置标注红色虚线和各曲线的 NMI 数值。

---

### 特征提取函数（第 534~628 行）

#### `extract_and_reduce_features`
提取 **ali/train** 上的 1280 维特征。

> **重要说明：** 此函数在修复后的代码中**不再被 `main()` 调用**。原因是贪心通道选择属于“模型选择”，应在验证集上进行，而非训练集。在训练集上做通道选择会导致严重的过拟合问题（训练 NMI 虚高到 1.0，详见 change.md）。函数本身保留以供其他用途。

#### `_extract_features`（通用版）
接收数据目录和描述文字作为参数，加载 OriginalModel，提取该目录下所有图片的 1280 维特征和标签。

#### `extract_val_features`、`extract_test_features`、`extract_megvii_test_features`
分别调用 `_extract_features` 在 Ali 验证集、Ali 测试集、Megvii 测试集上提取特征，用于评估通道选择的泛化效果。

---

### 主程序 `__main__`（第 687 行至文件末尾）

#### 配置参数

```python
MAX_K = 128              # 最多选 128 个通道
SCORE_N_SAMPLES = 3000   # 采样 3000 个样本进行打分
CHECKPOINTS = [32, 64, 128]  # 在这些维度记录 NMI
USE_GREEDY = True        # 使用贪心法（而非逐通道打分法）
USE_PARALLEL = True      # 使用多进程并行加速
N_WORKERS = 20           # 并行进程数
```

#### 执行流程

1. **提取特征**：依次在验证集、Ali 测试集、Megvii 测试集上提取 1280 维特征。
   > **注意：修复后不再提取训练集特征。** 通道选择属于「模型选择」，应在验证集上进行。原来在训练集上做通道选择会导致过拟合：模型本身就是在训练集上训练的，提取的训练特征对训练样本的区分度极高，贪心选择 60～80 个通道后训练 NMI 就会饱和到 1.0，此后的选择退化为随机选取。
2. **通道选择**：使用贪心法（多进程并行）在**验证集**上进行通道选择，贪心返回的 `nmi_curve` 直接就是验证集 NMI 曲线
3. **评估泛化**：用选出的通道在 Ali 测试集、Megvii 测试集上分别计算逐步 NMI 曲线
4. **画图**：画出三条折线的 NMI 曲线图并保存
5. **打印结果**：输出各 checkpoint 处三个数据集的 NMI
6. **保存选择器**：将选择结果保存到 `nmi_channel_selector.pkl`
7. **评估效果**：在验证集上用 `evaluate_channel_selection_nmi` 评估，在 k=128 时画 t-SNE 图
