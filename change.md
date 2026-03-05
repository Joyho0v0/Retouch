# 修改记录

## 2026-03-04：修复 ChannelSelect.py 贪心通道选择在训练集上过拟合导致 NMI 虚高到 1.0 的问题

### 问题现象

运行 `ChannelSelect.py` 进行贪心通道选择时，进度条中显示的 NMI 在约 60~80 个通道后就饱和到了 **1.0000**，看起来不太现实。

### 根本原因：通道选择过程中的"训练集过拟合"

贪心通道选择的原始代码流程如下：

1. **调用 `extract_and_reduce_features()` 提取 `ali/train`（训练集）的 1280 维特征**
2. 将训练特征传入 `greedy_select_channels_parallel(features=features, labels=labels, ...)`
3. 贪心循环中，对每个候选通道计算 **训练集上的 KMeans-NMI** 作为评分依据
4. 选择使训练集 NMI 最高的通道加入已选集合

**问题出在第 1~3 步**：OriginalModel 本身就是在 `ali/train` 上训练的，所以它提取的训练集特征对训练样本的区分度**极高**（模型已经"背住"了训练数据的模式）。当贪心法在训练特征上做通道选择时：

- **前 60~80 轮**：NMI 快速上升到接近 1.0，因为少量精选通道就足以在高区分度的训练特征空间中完美分开 1400 个训练样本（700 正/700 负）
- **第 80 轮之后**：NMI 已经饱和为 1.0（KMeans 完美聚类），**此后每个剩余通道加入后 NMI 都是 1.0**，贪心法无法区分通道的好坏，退化为随机选取
- **最终的 128 通道**：前 ~80 个是精心选出的，后 ~48 个本质上是随机的

这就类似于**训练准确率达到 100%**——并不代表模型真的完美，只是在训练数据上过拟合了。训练集 NMI=1.0 是一个**虚高的、无意义的指标**，既不能反映通道质量，也不能指导后续选择。

### 修复方案

**将贪心通道选择的评分数据从训练集改为验证集（`ali/val`）。**

通道选择在机器学习流程中属于**模型选择（Model Selection）**范畴，标准做法是在**验证集**上进行，而不是训练集：

| 阶段 | 数据集 | 说明 |
|------|--------|------|
| 模型训练 | `ali/train` | 已完成（OriginalModel） |
| **通道选择** | **`ali/val`** | **模型选择，应使用验证集** |
| 最终评估 | `ali/test`、`megvii/test` | 完全未参与选择过程的测试集 |

这样做的好处：
- 验证集特征没有被模型"背住"，NMI 不会虚高到 1.0
- 贪心算法在整个 128 步中都能有效区分通道好坏
- 进度条中的 NMI 真实反映通道的泛化能力
- 测试集完全不参与选择过程，保持评估的独立性

### 具体代码改动

#### ChannelSelect.py

1. **移除 `extract_and_reduce_features()` 的调用**（不再需要提取训练集特征）

   ```python
   # 修改前
   features, labels = extract_and_reduce_features()             # 训练集特征
   val_features, val_labels = extract_val_features()

   # 修改后
   val_features, val_labels = extract_val_features()             # 验证集特征，用于通道选择
   ```

2. **贪心函数传入验证集特征**

   ```python
   # 修改前
   greedy_select_channels_parallel(features=features, labels=labels, ...)

   # 修改后
   greedy_select_channels_parallel(features=val_features, labels=val_labels, ...)
   ```

3. **移除冗余的验证集 NMI 曲线重算**：贪心函数返回的 `nmi_curve` 现在就是验证集曲线，无需再单独重算

4. **移除训练集 NMI 的打印和保存**：`nmi_curve_train` 和 `checkpoint_results_train` 不再存在

5. **非贪心分支也改为使用验证集**：`select_top_k_channels_by_kmeans_nmi` 也传入 `val_features, val_labels`

#### RandomSelect.py

1. **原始模型（1280通道）NMI 改为只计算一次**：将 `orig_nmi_val/test/meg` 的计算从搜索循环内部移到循环前，后续画图直接复用
