# 问题分析与解决方案（第二轮）

## 一、实验数据对比

### V1（随机初始化 backbone + 通道选择）
| k | val_acc | ali_test | megvii_test |
|---|---|---|---|
| -1 | 0.9475 | 0.950 | 0.744 |
| 32 | 0.93 | 0.915 | **0.781** |

### 当前版本（预训练 backbone + 两阶段训练）
| k | ali_test | megvii_test |
|---|---|---|
| -1 | 0.950 | 0.744 |
| 32 | **0.945** | 0.752 |

对比结论：
- 源域 ali：当前版本明显好于 V1（0.945 vs 0.915），接近原模型（0.950）✓
- 跨域 megvii：当前版本比 V1 低（0.752 vs 0.781）✗
- 跨域 megvii：当前版本仍然比原模型好（0.752 vs 0.744）✓

---

## 二、对你的猜想的点评

### 猜想 1："上一个版本由于backbone参数随机初始化，造就了实验结果的偶然性"

**部分正确，但不是纯粹的偶然。** 随机初始化 backbone 确实引入了不确定性，但它之所以跨域效果好，本质是一种"被迫正则化"：
- 随机 backbone 让模型从零学习，只能通过 32 个通道去理解数据
- 极低的模型容量（随机特征 + 少通道）迫使模型学到更简单、更本质的模式
- 简单模式通常更具泛化性，所以跨域效果好
- 代价是源域准确率也下降了（0.915 vs 0.950）

这不完全是偶然——它是一种极端正则化带来的副作用。但这种方式不稳定（换个随机种子结果可能差很多）。

### 猜想 2："数据集分阶段"

**思路有道理，但实现复杂、收益不确定。** 你的核心直觉是对的：Phase 2 需要"新鲜信息"来推动模型继续学习。但存在几个问题：
- OriginalModel.pth 已经在全部 ali/train 上训练过了，backbone 已经"见过"所有数据
- 重新训练 OriginalModel 只用部分数据 → 得到更弱的基模型 → 通道选择质量也会下降
- 分割数据后每个阶段的数据量都变少，可能伤害最终效果

有一种更简单有效的方式可以达到类似效果（见方案部分）。

---

## 三、真正的问题根源

### 根因 1：Phase 2 完全无效——模型已收敛，早停立即触发

从 epoch_log_k32.csv 可以看到：
- Phase 1（freeze）：epoch 4 时 val_acc 就达到 0.9525
- Phase 2（finetune）：epoch 6-10 的 val_acc 在 0.94-0.95 之间波动，始终没超过 0.9525
- 在 epoch 10 时触发 patience=5 早停

原因：**预训练 backbone 的特征已经非常适合 ali 分类任务**。FC head 在 Phase 1 就能达到很好的效果。Phase 2 解冻 backbone 后，微小的 backbone 变化无法改善已经很好的 ali 准确率，但 patience 只有 5，backbone 还没来得及调整就被停掉了。

### 根因 2：训练数据增强太弱

当前 train.py 的 get_transforms() 只有：
- Resize(256) + CenterCrop(224) + RandomRotation(10)

缺少关键的增强手段：
- 没有 RandomHorizontalFlip → 模型依赖于图片方向
- 没有 ColorJitter → 模型依赖于源域的颜色/亮度分布
- 没有 RandomErasing → 模型依赖于局部细节而非整体模式

这些缺失导致模型过度拟合到 ali 数据集的视觉特性上，跨域到 megvii 时性能下降。

### 根因 3：正则化不够强

- Dropout 只有 0.2，对于 k=32（仅 32 维特征）来说太小
- 没有 label smoothing → 模型过度自信 → 泛化差
- Phase 2 的 backbone_lr=1e-5 对于已收敛的模型来说偏大

---

## 四、解决方案

### 核心思路

不需要拆分数据集。用更好的方法让 Phase 2 真正起作用：
1. **更强的数据增强** → 让模型学到与域无关的特征
2. **Label smoothing** → 防止模型对源域过度自信
3. **更高的 Dropout** → 在少通道条件下加强正则化
4. **Phase 2 重新计数 patience** → 给 backbone 微调足够的时间
5. **更小的 backbone_lr** → 防止破坏预训练特征
6. **更大的 Phase 2 patience** → 不要太快放弃

### 具体改动：newtrain.py

| 改动项 | 之前 | 之后 | 原因 |
|---|---|---|---|
| 训练增强 | Resize+CenterCrop+Rotation(10) | 加 RandomHorizontalFlip、ColorJitter、RandomGrayscale、RandomErasing | 减少对源域视觉特性的过拟合|
| Dropout | 0.2（固定） | k<=64 时用 0.4，k>64 时用 0.3 | 少通道时需要更强正则化 |
| Label smoothing | 无 | 0.1 | 防止过度自信 |
| backbone_lr | 1e-5 | 1e-6 | 更温和的 backbone 调整 |
| Phase 2 patience | 5（与 Phase 1 共用计数） | 15（Phase 2 开始时重置） | 给 backbone 微调足够时间 |
| Phase 1 patience | 无（不做 early stop） | 不变 | Phase 1 快速收敛不需要早停 |

### 不改动的文件
- train.py：保持原样（原模型训练不受影响）
- ChannelSelect.py：通道选择逻辑不变
- newtest.py：测试逻辑不变

### 预期效果
- 源域 ali：保持 0.94-0.95 水平（因为 backbone 仍加载预训练权重）
- 跨域 megvii：预期提升到 0.76-0.80（更强增强 + label smoothing 减少源域过拟合）
- Phase 2 不再立即早停，backbone 有时间做有意义的微调

---

## 第三轮修改：NMI 评估数据泄漏修复

### 问题发现
`results/nmi_curve.png` 中 NMI 从 k=10 开始就达到 1.0（完美分数），明显异常。

### 根因分析
`ChannelSelect.py` 中存在三重数据泄漏：

1. **特征提取**：`extract_and_reduce_features()` 在 `ali/train` 上提取特征
2. **通道选择**：贪心法在 **同一批训练特征** 上最大化 KMeans-NMI
3. **效果评估**：NMI 曲线绘制 + checkpoint 评估 + t-SNE 可视化，**仍然使用同一批训练特征**

OriginalModel 本身就在 `ali/train` 上训练到 ~95% 准确率，提取的 1280 维特征对训练样本已高度可分。贪心法再专门挑能让这批数据 NMI 最高的通道 → 只需 10 个通道就能让 KMeans 在训练数据上完美聚类 → NMI=1.0。

**通道选择用训练集是合理的**（选通道本身就是"训练"过程），但 **评估必须用模型未见过的数据**，否则无法反映泛化能力。

### 修改内容（ChannelSelect.py）

#### 1. 新增 `extract_val_features()` 函数
在 `extract_and_reduce_features()` 之后新增函数，从 `dataset/ali/val` 提取特征：
- 加载同一个 OriginalModel
- 在验证集上前向传播得到 1280 维特征
- 返回 `(X_val, y_val)`

#### 2. `__main__` 中同时提取训练集和验证集特征
```python
features, labels = extract_and_reduce_features()       # 训练集特征，用于通道选择
val_features, val_labels = extract_val_features()       # 验证集特征，用于评估
```
通道选择仍基于训练集特征（这是合理的），但所有评估改为验证集。

#### 3. 贪心选择后，在验证集上重新计算 NMI 曲线
选通道完成后，遍历 k=1~256，在 **验证集特征** 上重新计算每个 k 的 subspace KMeans-NMI，生成新的 `val_nmi_curve`，用于画图和 checkpoint 报告。

#### 4. NMI 曲线图和 checkpoint 打印改为验证集结果
- `nmi_curve.png` 画的是验证集 NMI 曲线
- 控制台同时打印训练集和验证集的 checkpoint NMI，方便对比

#### 5. pkl 中同时保存训练集和验证集 NMI
```
nmi_curve         → 验证集 NMI 曲线（主要参考）
nmi_curve_train   → 训练集 NMI 曲线（仅供对比）
checkpoint_results       → 验证集 checkpoint NMI
checkpoint_results_train → 训练集 checkpoint NMI
```

#### 6. 评估段改为在验证集上评估
`evaluate_channel_selection_nmi()` 和 t-SNE 可视化均改为使用 `val_features` 和 `val_labels`。

### 未改动的部分
- 通道选择算法本身（贪心法仍基于训练集特征，这是正确的）
- `extract_and_reduce_features()` 函数体不变
- 其他所有文件不变