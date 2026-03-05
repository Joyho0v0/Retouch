# 代码说明

## EfficientNet_B0.py
1. [EfficientNet_B0.py](EfficientNet_B0.py#L1-L33) 定义了两个工具函数：`drop_connect` 在训练时随机掐断残差支路，`SqueezeExcite` 通过全局池化和 1×1 卷积给通道重新加权。
2. [EfficientNet_B0.py](EfficientNet_B0.py#L35-L74) 描述了 EfficientNet 的核心 `MBConvBlock`：可选的展开卷积、深度可分离卷积、SE 模块、投影卷积以及带 drop connect 的残差相加。
3. [EfficientNet_B0.py](EfficientNet_B0.py#L77-L146) 组装主干：包括输入 stem、每层的配置表 `_build_blocks()`、用于升维的 head，以及把特征压到 1×1 的自适应池化。
4. [EfficientNet_B0.py](EfficientNet_B0.py#L148-L166) 负责权重初始化并实现 `forward()`，它调用 `forward_features()`（stem → blocks → head → 可选池化/展平）后接 dropout+线性分类器输出类别概率。
5. [EfficientNet_B0.py](EfficientNet_B0.py#L167-L186) 提供 `create_feature_extractor()` 和一个简易 `FeatureExtractor` 包装，方便其它脚本直接拿到池化后的特征图。

## train.py
1. [train.py](train.py#L1-L35) 导入依赖，定义训练/验证的图像增强，并通过 `build_model()` 实例化二分类版本的 EfficientNet-B0。
2. [train.py](train.py#L37-L83) 给出两个循环：`train_one_epoch()` 负责前向、反向、优化和统计训练指标，`evaluate()` 在关闭梯度的情况下计算验证集损失与准确率。
3. [train.py](train.py#L86-L135) 从 `dataset/train`、`dataset/val` 读取数据，构建 DataLoader，设置设备、损失函数、优化器与早停计数，最多训练 100 轮并把最佳模型权重存到 `results/model.pth`。

## FeatureExtractor.py
1. [FeatureExtractor.py](FeatureExtractor.py#L1-L25) 复用 `train.py` 的函数和 `EfficientNet_B0.FeatureExtractor`，加载训练完的模型并暴露其池化特征。
2. [FeatureExtractor.py](FeatureExtractor.py#L26-L41) 在没有数据增强的情况下遍历整个训练集，把每个 `[B,1280,1,1]` 特征拉平成 1280 维向量并与标签一起累积。
3. [FeatureExtractor.py](FeatureExtractor.py#L43-L59) 对全部特征做 128 维 PCA，打印方差信息，并把降维后的特征、标签以及 PCA 模型分别保存到 `results/` 目录。

## newTrain.py
1. [newTrain.py](newTrain.py#L1-L26) 加载同样的工具函数与 scikit-learn 组件，继而读入 `results/model.pth` 权重及 `dataset/test` 的 DataLoader。
2. [newTrain.py](newTrain.py#L27-L40) 通过 `model.forward_features(..., pool=True, flatten=False)` 提取 1280 维验证特征，再展平供后续处理。
3. [newTrain.py](newTrain.py#L41-L62) 使用之前保存的 PCA 模型压缩验证特征，基于训练集的 PCA 结果拟合逻辑回归，并在验证集上评估准确率。

## 代码间的关系
1. `train.py` 负责端到端训练 EfficientNet，并把最佳权重保存到 `results/model.pth`。
2. `FeatureExtractor.py` 重新加载该权重，利用同一个主干抽取训练集特征，并输出 PCA 压缩后的特征矩阵、标签与 PCA 模型。
3. `newTrain.py` 同时消耗 EfficientNet 的权重和 PCA 产物，在另一份数据集上用逻辑回归验证压缩特征的区分能力。
4. `EfficientNet_B0.py` 是整个流程的核心：既提供训练时的分类器，也提供特征抽取接口供 PCA 与后续的浅层模型使用。
