## 代码逐行讲解与整体关系说明

下面依次对 EfficientNet_B0.py、train.py、FeatureExtractor.py、newTrain.py 进行逐行/逐块说明，最后总结四个脚本之间的整体流程和关系。

---

## 一、EfficientNet_B0.py：模型结构与特征提取

这个文件主要实现了一个“可读性较高的 EfficientNet-B0”，并顺便内置了一个特征提取器封装。

```python
"""Readable EfficientNet-B0 re-implementation without fancy syntax."""
```
- 文件的模块级文档字符串，说明这是一个“可读版本”的 EfficientNet-B0 实现。

```python
import torch
from torch import nn
```
- 导入 PyTorch 的核心包和神经网络模块 nn。

```python
def drop_connect(x, drop_prob, training):
	keep_prob = 1.0 - drop_prob
	if not training or drop_prob == 0.0:
		return x
	random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
	random_tensor = random_tensor.floor()
	return x * random_tensor / keep_prob
```
- 定义 drop_connect 函数，用于在残差连接中随机“掐断”一部分样本的连接（Stochastic Depth 思想）。
- `keep_prob`：保持连接的概率，等于 1 - drop_prob。
- 如果当前不是训练模式，或者 drop_prob 为 0，就直接返回原始输入 x，不做任何随机掐断。
- `torch.rand(x.shape[0], 1, 1, 1)`：对 batch 维度按样本生成随机数，同一张图上所有通道/空间位置共享一个随机值。
- `+ keep_prob` 再 `floor()` 相当于以 keep_prob 的概率得到 1，以 (1-keep_prob) 的概率得到 0。
- 最后用这个 0/1 掩码乘以特征，再除以 keep_prob 做期望值的缩放，使整体期望幅度不变。

```python
class SqueezeExcite(nn.Module):
	def __init__(self, in_channels, se_ratio=0.25):
		super().__init__()
		reduced_channels = max(1, int(in_channels * se_ratio))
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
		self.act = nn.SiLU(inplace=True)
		self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
		self.gate = nn.Sigmoid()

	def forward(self, x):
		scale = self.pool(x)
		scale = self.reduce(scale)
		scale = self.act(scale)
		scale = self.expand(scale)
		scale = self.gate(scale)
		return x * scale
```
- 定义 SE（Squeeze-and-Excitation）注意力模块。
- `reduced_channels`：把通道数先降到 in_channels * se_ratio（默认 1/4），至少为 1。
- `self.pool`：自适应平均池化到 1×1，实现全局信息的“squeeze”。
- `self.reduce`：1×1 卷积实现通道降维。
- `self.act`：SiLU 激活函数。
- `self.expand`：1×1 卷积把通道升回原通道数。
- `self.gate`：Sigmoid，把尺度控制在 (0,1)。
- forward 中：对输入 x 先全局池化，再降维+激活+升维+Sigmoid 得到每个通道的缩放系数 scale；最后 `x * scale` 做通道注意力。

```python
class MBConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size, drop_connect_rate):
		super().__init__()
		self.use_residual = stride == 1 and in_channels == out_channels
		self.drop_connect_rate = drop_connect_rate
		hidden_dim = in_channels * expand_ratio
		padding = kernel_size // 2

		layers = []
		if expand_ratio != 1:
			layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
			layers.append(nn.BatchNorm2d(hidden_dim))
			layers.append(nn.SiLU(inplace=True))

		layers.append(
			nn.Conv2d(
				hidden_dim,
				hidden_dim,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
				groups=hidden_dim,
				bias=False,
			)
		)
		layers.append(nn.BatchNorm2d(hidden_dim))
		layers.append(nn.SiLU(inplace=True))
		layers.append(SqueezeExcite(hidden_dim))
		layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
		layers.append(nn.BatchNorm2d(out_channels))

		self.block = nn.Sequential(*layers)

	def forward(self, x):
		out = self.block(x)
		if self.use_residual:。
			out = drop_connect(out, self.drop_connect_rate, self.training)
			out = out + x
		return out
```
- MBConvBlock：Mobile Inverted Bottleneck 卷积块，是 EfficientNet 的基本单元。
- `self.use_residual`：只有 stride=1 且 in_channels == out_channels 时，才能做残差相加。
- `hidden_dim`：扩展后的中间通道数，= in_channels * expand_ratio。
- 若 `expand_ratio != 1`，先做一个 1×1 卷积升维（Point-wise 卷积）+ BN + SiLU。
- 然后加入 Depthwise 卷积（groups=hidden_dim），带 stride 和 padding。
- 再做 BN + SiLU，接上 SqueezeExcite 注意力模块。
- 最后用 1×1 卷积降回 out_channels，并 BN。
- `self.block` 用 nn.Sequential 把所有层串起来。
- forward：先过 `self.block` 得到 out，如果允许残差，就先调用 drop_connect，再与输入 x 相加，最后返回。

```python
class EfficientNetB0(nn.Module):
	def __init__(self, num_classes=1000, drop_connect_rate=0.2):
		super().__init__()
		self.drop_connect_rate = drop_connect_rate

		self.stem = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.SiLU(inplace=True),
		)

		self.block_settings = [
			# expand, out, repeats, stride, kernel
			(1, 16, 1, 1, 3),
			(6, 24, 2, 2, 3),
			(6, 40, 2, 2, 5),
			(6, 80, 3, 2, 3),
			(6, 112, 3, 1, 5),
			(6, 192, 4, 2, 5),
			(6, 320, 1, 1, 3),
		]

		self.blocks = self._build_blocks()
		self.head = nn.Sequential(
			nn.Conv2d(320, 1280, kernel_size=1, bias=False),
			nn.BatchNorm2d(1280),
			nn.SiLU(inplace=True),
		)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1280, num_classes),
		)

		self._init_weights()
```
- EfficientNetB0 主干结构类。
- `num_classes`：最后分类头的类别数，在本项目中为 2（二分类）。
- `self.stem`：最前面的卷积 stem：3→32 通道，stride=2，把输入图片分辨率减半。
- `self.block_settings`：定义每个 stage 的配置：扩展比例、输出通道数、重复次数、stride、卷积核大小。
- `self.blocks = self._build_blocks()`：根据 block_settings 构建整个 backbone。
- `self.head`：在 backbone 末尾再做一次 320→1280 的 1×1 卷积 + BN + SiLU。
- `self.avgpool`：全局平均池化到 1×1。
- `self.classifier`：Dropout(0.2) + 全连接层，把 1280 维特征映射到 num_classes 分类。
- 最后调用 `_init_weights()` 对网络权重进行初始化。

```python
	def forward_features(self, x, pool=True, flatten=True):
		x = self.stem(x)
		x = self.blocks(x)
		x = self.head(x)
		if pool:
			x = self.avgpool(x)
		if flatten:
			x = torch.flatten(x, 1)
		return x
```
- `forward_features`：只前向计算到“特征层”，可控制是否做 avgpool 和 flatten。
- 这样一来：
  - `pool=True, flatten=True`：输出 [B, 1280] 的一维特征（适合给 classifier）。
  - `pool=True, flatten=False`：输出 [B, 1280, 1, 1] 的特征图（方便 reshape）。
  - `pool=False`：保留空间分辨率的 feature map。

```python
	def _build_blocks(self):
		blocks = []
		current_channels = 32
		total_blocks = sum(setting[2] for setting in self.block_settings)
		built_blocks = 0

		for setting in self.block_settings:
			expand, out_channels, repeats, stride, kernel_size = setting
			for repeat in range(repeats):
				block_stride = stride if repeat == 0 else 1
				drop_rate = self.drop_connect_rate * built_blocks / total_blocks
				block = MBConvBlock(
					in_channels=current_channels,
					out_channels=out_channels,
					stride=block_stride,
					expand_ratio=expand,
					kernel_size=kernel_size,
					drop_connect_rate=drop_rate,
				)
				blocks.append(block)
				current_channels = out_channels
				built_blocks += 1

		return nn.Sequential(*blocks)
```
- `_build_blocks` 根据配置循环构造 MBConvBlock 列表。
- `current_channels`：当前 block 输入通道，初始是 stem 输出 32。
- `total_blocks`：计算所有 stage 的总 block 数，用于按层数线性增加 drop_connect_rate。
- 每个 setting 里有 repeats 次：
  - 第一个 repeat 用给定 stride，其余 repeat stride=1。
  - `drop_rate` 随着 built_blocks 递增（前面的块 drop 小，后面的块 drop 大）。
- 将所有 block 放进 nn.Sequential 返回。

```python
	def _init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.01)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
```
- `_init_weights`：对各类层按照常见规则做初始化：
  - Conv2d：Kaiming Normal；bias 置 0。
  - BatchNorm2d：权重初始化为 1，偏置为 0。
  - Linear：高斯分布初始化权重，偏置 0。

```python
	def forward(self, x):
		x = self.forward_features(x, pool=True, flatten=True)
		x = self.classifier(x)
		return x

	def create_feature_extractor(self, pool=True, flatten=True):
		return FeatureExtractor(self, pool=pool, flatten=flatten)
```
- `forward`：完整前向网络：先抽取特征，再通过分类头输出 logits。
- `create_feature_extractor`：基于当前 backbone 创建一个 FeatureExtractor 封装，方便后面对特征进行提取，而不必自己手动调用 forward_features。

```python
class FeatureExtractor(nn.Module):
	def __init__(self, backbone, pool=True, flatten=True):
		super().__init__()
		self.backbone = backbone
		self.pool = pool
		self.flatten = flatten

	def forward(self, x):
		return self.backbone.forward_features(x, pool=self.pool, flatten=self.flatten)
```
- FeatureExtractor：一个简单包装类，把 EfficientNetB0 的 forward_features 封装成独立模块。
- 你可以指定是否 pool、是否 flatten，从而控制输出特征的形状。

```python
def efficientnet_b0(num_classes=1000, drop_connect_rate=0.2):
	return EfficientNetB0(num_classes=num_classes, drop_connect_rate=drop_connect_rate)


__all__ = ["EfficientNetB0", "efficientnet_b0", "FeatureExtractor"]
```
- `efficientnet_b0`：一个工厂函数，便于外部调用构建模型。
- `__all__`：限定 from EfficientNet_B0 import * 时导出的对象。

小结：
- EfficientNet_B0.py 定义了整个 EfficientNet-B0 主干网络结构和一个 FeatureExtractor 包装类。
- 这个模型既可以用于直接分类训练（train.py 中用），也可以仅用于抽取中间特征（FeatureExtractor.py 与 newTrain.py 中用）。

---

## 二、train.py：训练原始 EfficientNet-B0 分类模型

这个文件完成从数据加载、数据增强、模型构建，到训练循环与早停保存 best model 的完整流程。

```python
import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
from torchvision import datasets, transforms
from EfficientNet_B0 import EfficientNetB0
import numpy as np
```
- 导入训练相关库：os、torch、tqdm、nn、optim、DataLoader、torchvision 数据集与变换。
- 从 EfficientNet_B0 文件中导入 EfficientNetB0 模型。
- numpy 在本文件中其实没有被使用，但导入了（可视为冗余）。

```python
def get_transforms():
	train_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return train_transform, val_transform
```
- 定义数据增强 / 预处理：
  - 训练集：先 Resize 到 256，CenterCrop 到 224，再随机旋转 10°，转换为 Tensor 并用 ImageNet 均值方差做标准化。
  - 验证集：同样的 Resize + CenterCrop + ToTensor + Normalize，但不做随机旋转（无数据增强）。
- 返回一对 transform，用于后续构建数据集。

```python
def build_model(num_classes):
	# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
	# in_features = model.classifier[1].in_features
	# model.classifier[1] = nn.Linear(in_features, num_classes)
	# return model
	return EfficientNetB0(num_classes=num_classes)
```
- 构建模型：
  - 上面注释掉的部分是使用 torchvision 内置 EfficientNet-B0 的版本。
  - 实际使用的是自己在 EfficientNet_B0.py 中实现的 EfficientNetB0，输出维度为 num_classes。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in tqdm(loader, desc="Train", leave=False):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct += torch.sum(preds == labels).item()
		total += labels.size(0)

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc
```
- `train_one_epoch`：在一个 epoch 内对训练集循环：
  - `model.train()`：设置为训练模式（启用 dropout / BN 的训练行为）。
  - 遍历 DataLoader：送入图片和标签到 device。
  - 清零梯度，前向得到输出，计算 loss，反向传播，optimizer.step() 更新参数。
  - `running_loss` 累加；通过 `torch.max` 计算预测类别，与标签比较计算正确数量。
  - 最终返回该 epoch 的平均 loss 和准确率。

```python
def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Val", leave=False):
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += torch.sum(preds == labels).item()
			total += labels.size(0)

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc
```
- `evaluate`：验证/测试过程：
  - `model.eval()`：切换到评估模式（关闭 dropout，BN 使用滑动均值等）。
  - `torch.no_grad()`：不计算梯度以节省显存和计算。
  - 其他流程和 train_one_epoch 类似，只是不做反向传播和优化。

```python
def main():
	base_dir = "./dataset"
	train_dir = os.path.join(base_dir, "train")
	val_dir = os.path.join(base_dir, "val")

	train_transform, val_transform = get_transforms()

	train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
	val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = build_model(num_classes=2)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	epochs = 100
	patience = 10
	no_improve_epochs = 0
	best_acc = 0.0
	save_path = "./results/model.pth"
```
- main 函数整体训练入口。
- `base_dir` 指向 dataset 目录，下有 train / val 子目录，按 ImageFolder 的类文件夹结构组织。
- 使用 `get_transforms` 得到训练/验证 transforms，再构建 ImageFolder 数据集。
- DataLoader：
  - batch_size=16，训练集 shuffle=True，验证集 shuffle=False。
- 根据是否有 CUDA 选择 device。
- 使用 `build_model(2)` 构建二分类 EfficientNetB0 模型并转到 device。
- 损失函数：交叉熵；优化器：Adam(lr=1e-4)。
- 训练超参数：最多 100 个 epoch，patience=10 表示若验证准确率 10 个 epoch 不提升则早停。
- `save_path`：最佳模型权重保存到 ./results/model.pth。

```python
	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)

		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), save_path)
			no_improve_epochs = 0
		else:
			no_improve_epochs += 1

		print("Epoch {}/{}".format(epoch, epochs))
		print("Train Loss: {:.4f}, Train Acc: {:.4f}".format(train_loss, train_acc))
		print("Val Loss: {:.4f}, Val Acc: {:.4f}".format(val_loss, val_acc))
		print("Best Acc: {:.4f}".format(best_acc))
		print("No improve epochs: {} / {}".format(no_improve_epochs, patience))

		if no_improve_epochs >= patience:
			print("Validation accuracy did not improve for {} epochs. Early stopping.".format(patience))
			break

	print("Training finished. Best model saved to:")
	print(save_path)
```
- 循环每个 epoch：先训练一个 epoch，再在验证集上评估。
- 如果当前 val_acc 优于历史 best_acc，则：
  - 更新 best_acc。
  - `torch.save(model.state_dict(), save_path)`：保存当前最优权重。
  - 重置 no_improve_epochs = 0。
- 否则，no_improve_epochs 自增 1。
- 打印当前 epoch 的指标和 best_acc、早停计数。
- 当连续 no_improve_epochs ≥ patience 时触发早停，break 退出训练。
- 最后打印提示 best model 保存路径。

```python
if __name__ == "__main__":
	main()
```
- 作为脚本运行时，执行 main()，启动训练流程。

小结：
- train.py 的作用：**用原始图像直接训练一个二分类 EfficientNet-B0 模型，并把验证集表现最好的权重保存到 ./results/model.pth。**

---

## 三、FeatureExtractor.py：从训练好的模型提取特征并做 PCA 降维

这个文件依赖 train.py 中的构建模型和 transforms 的逻辑，对训练集图片提取中间特征（1280 维），然后用 PCA 降到 128 维并保存下来。

```python
from train import *
from EfficientNet_B0 import FeatureExtractor as EfficientNetFeatureExtractor
```
- `from train import *`：导入 train.py 中的所有公开对象（包括 get_transforms、build_model 等）。
- 从 EfficientNet_B0 中导入 FeatureExtractor 类，并起别名 EfficientNetFeatureExtractor，避免与本文件名混淆。

```python
# 特征提取+PCA降维
def extract_and_reduce_features():
	base_dir = "./dataset"
	train_dir = os.path.join(base_dir,"train")
```
- 定义核心函数 extract_and_reduce_features，用于：
  1. 提取训练集特征；
  2. 对特征做 PCA 降维；
  3. 保存降维后的特征和标签，以及 PCA 模型本身。
- `train_dir` 指向训练集目录。

```python
	# 使用验证转换（无数据增强）
	_, val_transform = get_transforms()
	train_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
	train_loader = DataLoader(train_dataset,batch_size=16,shuffle=False,num_workers=2)
```
- 为了保证特征一致性，这里使用 get_transforms 中“验证集”的 transform（即不做随机增强），只做 Resize + CenterCrop + 标准化。
- 使用 ImageFolder 构建训练集 Dataset，再用 DataLoader 封装：
  - batch_size=16；
  - shuffle=False（提特征通常不打乱，方便后续对齐）。

```python
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 加载训练好的模型
	model = build_model(num_classes=2)
	model.load_state_dict(torch.load("./results/model.pth",map_location=device))
	model = model.to(device)
	model.eval()
```
- 和 train.py 保持一致，构建同一个 EfficientNetB0(num_classes=2) 结构的模型。
- 从 ./results/model.pth 加载在 train.py 中训练得到的 best 权重。
- 把模型放到对应设备并设置为 eval() 模式。

```python
	# 构建特征提取器：去掉classifier,保留feature + avgpool
	extractor = EfficientNetFeatureExtractor(model, pool=True, flatten=False)
```
- 使用 EfficientNetFeatureExtractor 封装模型：
  - `pool=True`：会做 avgpool，得到 [B, 1280, 1, 1]；
  - `flatten=False`：不做 flatten，保持 4D 形状，便于后面手动 `.view`。
- 这一行逻辑上意味着“只保留 backbone 与 avgpool，不走 classifier 线性层”，用于获得中间特征而不是最终 logits。

```python
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
	y = np.concatenate(all_labels, axis=0)	#[N,]

	print(f"Feature matrix shape:{X.shape}")
```
- 初始化两个 list 用来保存所有 batch 的特征和标签。
- 在 no_grad 下遍历训练集：
  - 把图片丢到 GPU/CPU 上，经过 extractor 得到特征图 [B, 1280, 1, 1]；
  - `view(B, -1)` 把其拉平成 [B, 1280]；
  - 转为 numpy 并 append 到 all_features 中，labels 同样转 numpy 存入 all_labels。
- `np.concatenate` 把列表中的 batch 拼成完整矩阵：
  - X: [N, 1280]，N 为训练集中所有样本数；
  - y: [N]，对应标签。

```python
	# PCA降维
	from sklearn.decomposition import PCA
	print("Performing PCA to 128 demensions....")
	pca = PCA(n_components=128, svd_solver="full")
	X_reduced = pca.fit_transform(X)
	print(f"Reduced feature shape: {X_reduced.shape}")
	print(f"Explained variance ratio (top 5): {pca.explained_variance_ratio_[:5]}")
	print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
```
- 导入 sklearn 的 PCA。
- 构造 PCA 对象：
  - n_components=128：降到 128 维；
  - svd_solver="full"：使用完整 SVD 算法。
- `fit_transform(X)`：在训练特征 X 上拟合 PCA 并做变换，得到降维后的 X_reduced [N, 128]。
- 打印降维后形状和前 5 个主成分方差占比以及总方差解释率。

```python
	# 保存结果
	save_dir = "./results"
	np.save(os.path.join(save_dir, "features_pca128.npy"), X_reduced)
	np.save(os.path.join(save_dir, "labels.npy"), y)

	# PCA模型
	import joblib
	joblib.dump(pca, os.path.join(save_dir, "pca_model.pth"))
	print("Features and PCA model saved.")
```
- 将降维后的特征和对应标签保存到 ./results 目录下：
  - features_pca128.npy：形状 [N, 128]；
  - labels.npy：形状 [N]。
- 使用 joblib.dump 保存训练好的 PCA 模型到 pca_model.pth（实际是 joblib 格式的文件）。
- 后续会在 newTrain.py 中加载这个 PCA 模型。

```python
if __name__ == "__main__":

	# 如果你想只运行训练，注释掉下面这行
	# main()
	
	# 如果你想提取特征，取消注释下面这行
	extract_and_reduce_features()
```
- 作为脚本运行时调用 extract_and_reduce_features()。
- 注释里提示：如果只想训练可以调用 main()（实际上这里注释掉了 main 的导入和调用，当前脚本只做特征+PCA）。

小结：
- FeatureExtractor.py 的作用：**使用 train.py 训练好的 EfficientNet-B0 模型，对训练集图片提取“1280 维特征”，做 PCA 降到 128 维，并把降维特征、标签和 PCA 模型保存到 ./results。**

---

## 四、newTrain.py：使用 PCA 特征 + 逻辑回归做评估

这个文件不再直接用神经网络最后的分类头，而是：
- 用训练好的 EfficientNet-B0 模型抽取“验证（这里是 test）集”的 1280 维特征；
- 用之前训练好的 PCA 模型把这些特征映射到 128 维；
- 再在训练集的 PCA 特征上训练 Logistic Regression 分类器，最后在验证集的 PCA 特征上评估分类效果。

```python
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from train import get_transforms, build_model  # 假设 train.py 在同目录
from torchvision import datasets, transforms
from EfficientNet_B0 import EfficientNetB0
from torch.utils.data import DataLoader
import joblib
```
- 导入：
  - torch、numpy、tqdm；
  - sklearn 的 LogisticRegression 和 accuracy_score；
  - 从 train.py 导入 get_transforms 与 build_model，以复用数据预处理和模型结构；
  - torchvision.datasets 用于加载 test 数据；
  - EfficientNet_B0 中的 EfficientNetB0（这里其实不直接用到，因为通过 build_model 取得同样结构即可）；
  - joblib 用于加载之前保存的 PCA 模型。

```python
def evaluate_pca_performance():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 1. 加载原始模型P（用于提取验证集特征）
	model = build_model(num_classes=2)
	model.load_state_dict(torch.load("./results/model.pth", map_location=device))
	model = model.to(device)
	model.eval()
```
- `evaluate_pca_performance`：完整评估函数。
- 首先与之前一致，构造一个 EfficientNetB0(num_classes=2) 的模型，加载训练好的 best 权重，放到 device 上并设为 eval()。

```python
	# 2. 加载验证集
	_, val_transform = get_transforms()
	val_dataset = datasets.ImageFolder("./dataset/test", transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
```
- 这里把 `./dataset/test` 当作“验证集”（或者测试集）来用。
- 同样使用 get_transforms 的验证 transform（无数据增强）。
- DataLoader：16 batch 大小，不打乱。

```python
	# 3. 提取验证集原始特征 (1280-dim)
	print("Extracting validation set features...")
	val_features_1280 = []
	val_labels = []
	with torch.no_grad():
		for images, labels in tqdm(val_loader, desc="Val Feature Extraction"):
			images = images.to(device)
			feats = model.forward_features(images, pool=True, flatten=False)
			feats = feats.view(feats.size(0), -1).cpu().numpy()
			val_features_1280.append(feats)
			val_labels.append(labels.numpy())
	X_val_1280 = np.concatenate(val_features_1280, axis=0)
	y_val = np.concatenate(val_labels, axis=0)
```
- 不再通过 FeatureExtractor 包装，而是直接调用 `model.forward_features(pool=True, flatten=False)` 得到 [B, 1280, 1, 1] 特征图。
- 然后 view 成 [B, 1280]，转 numpy，拼接成：
  - X_val_1280: 验证集的 1280 维特征矩阵；
  - y_val: 验证集标签。

```python
	# 4. 加载 PCA 模型，并 transform 验证集特征
	pca = joblib.load("./results/pca_model.pth")  # 注意：你保存的是 .pth，但 joblib 能读
	X_val_pca = pca.transform(X_val_1280)  # 关键：用训练集学的 PCA 变换验证集！
	print(f"Val PCA features shape: {X_val_pca.shape}")
```
- 使用 joblib.load 读取之前在 FeatureExtractor.py 中保存的 PCA 模型。
- 用这个 PCA 对验证集的 1280 维特征做变换（transform），得到 128 维的 `X_val_pca`。

```python
	# 5. 加载训练集 PCA 特征和标签（用于训练分类器）
	X_train_pca = np.load("./results/features_pca128.npy")
	y_train = np.load("./results/labels.npy")
```
- 从 ./results 加载之前保存好的：
  - 训练集 PCA 特征 X_train_pca；
  - 训练集标签 y_train。

```python
	# 6. 训练简单分类器（Logistic Regression）
	print("Training Logistic Regression on train PCA features...")
	clf = LogisticRegression(max_iter=1000, random_state=42)
	clf.fit(X_train_pca, y_train)
```
- 在训练集 PCA 特征上训练一个简单的线性分类器（逻辑回归）：
  - max_iter=1000：最多迭代 1000 步以保证收敛；
  - random_state=42：固定随机种子保证可复现。

```python
	# 7. 在验证集 PCA 特征上预测
	y_pred = clf.predict(X_val_pca)
	acc = accuracy_score(y_val, y_pred)
	
	print("\n" + "="*50)
	print(f"Final Evaluation on VALIDATION SET")
	print(f"PCA (128-dim) + Logistic Regression Accuracy: {acc:.4f} ({acc*100:.2f}%)")
	print("="*50)
```
- 用训练好的逻辑回归分类器，对验证集的 PCA 特征进行预测，得到预测标签 y_pred。
- 使用 accuracy_score 计算在验证/测试集上的准确率 acc。
- 打印最终评估结果：128 维 PCA 特征 + Logistic Regression 在验证集上的准确率。

```python
if __name__ == "__main__":
	evaluate_pca_performance()
```
- 作为脚本运行时，调用 evaluate_pca_performance() 完成整体评估流程。

小结：
- newTrain.py 的作用：**使用神经网络抽取的 PCA 特征作为输入，训练一个传统机器学习分类器（Logistic Regression），并在测试集上评估“PCA+LR”这一方案的性能。**

---

## 五、四个脚本之间的整体关系与流程

从功能上看，四个脚本构成了一个完整的“深度特征 + 传统分类器”的流程：

1. EfficientNet_B0.py
   - 定义了 EfficientNet-B0 模型结构（含 MBConvBlock、SE 注意力、drop_connect 等模块）。
   - 提供了 forward / forward_features 和 FeatureExtractor 封装，既可以用于直接端到端分类，也可以方便地抽取中间特征。

2. train.py
   - 使用 EfficientNetB0(num_classes=2) 在原始图像上进行“端到端的二分类训练”。
   - 使用 ImageFolder 加载 dataset/train 和 dataset/val，并做基础预处理与适度的数据增强。
   - 每个 epoch 训练后在验证集上评估，如果验证准确率提升就保存 best 模型到 ./results/model.pth。
   - 最终得到一个“训练好的深度模型 P”，其权重保存在 model.pth 中。

3. FeatureExtractor.py
   - 读取 train.py 训练好的 best 模型权重 model.pth。
   - 使用 FeatureExtractor 封装，把模型视作一个“特征提取网络”，提取训练集图片的 1280 维全局平均池化特征。
   - 对所有训练样本的特征组成矩阵 X [N, 1280]，对标签组成 y [N]。
   - 使用 PCA 将 1280 维特征降到 128 维，得到 X_reduced [N, 128]。
   - 将 X_reduced 保存为 features_pca128.npy，y 保存为 labels.npy，同时保存 PCA 模型为 pca_model.pth。
   - 这一步相当于“用深度网络做特征抽取，再用无监督的 PCA 做特征压缩”。

4. newTrain.py
   - 再次加载训练好的 EfficientNet-B0（model.pth），但此时只用来抽取验证/测试集的 1280 维特征。
   - 使用在训练集上 fit 出来的 PCA 模型（pca_model.pth）将验证集特征变换到 128 维。
   - 加载之前保存的训练集 PCA 特征和标签（features_pca128.npy、labels.npy），在这些特征上训练 Logistic Regression 分类器。
   - 用训练好的 Logistic Regression 对验证/测试集的 PCA 特征做预测，计算并打印准确率。
   - 这一步相当于“在深度特征基础上，使用传统机器学习模型做分类并评估效果”。

整体流程可以概括为：

1. 先运行 train.py：
   - 得到最优的 EfficientNet-B0 模型权重 model.pth。

2. 再运行 FeatureExtractor.py：
   - 使用 model.pth 对训练集抽取特征并做 PCA 降维，得到 features_pca128.npy、labels.npy 和 pca_model.pth。

3. 最后运行 newTrain.py：
   - 使用 model.pth 抽取测试集特征，用 pca_model.pth 做相同的 PCA 变换；
   - 在 features_pca128.npy / labels.npy 上训练 Logistic Regression；
   - 在测试集 PCA 特征上评估 Logistic Regression 的准确率。

从“做了什么事”的角度简单总结：

- EfficientNet_B0.py：**定义模型结构，提供特征提取接口。**
- train.py：**用原始图片训练这个模型做二分类，得到最优权重 model.pth。**
- FeatureExtractor.py：**用训练好的模型对训练集提取 1280 维特征并 PCA 到 128 维，保存特征和 PCA 模型。**
- newTrain.py：**在 PCA 特征上训练 Logistic Regression，并在测试集上评估 PCA+LR 的性能。**

如果你后续还希望，我可以在 ReadMe.md 中再画一个简单的流程图或伪代码，把数据流（图像 → CNN → 特征 → PCA → Logistic Regression）表示得更直观。

---

## 六、整体数据流程图

下面用一个简单的“数据流”示意，把四个脚本之间的信息传递关系串起来：

```text
	   ┌─────────────────────────────┐
	   │         原始图像数据        │
	   │  dataset/train, val, test  │
	   └─────────────┬──────────────┘
			     │
			     │(数据加载 + 预处理)
			     ▼
		     ┌───────────────┐
		     │   train.py    │
		     │ EfficientNetB0│
		     └───────┬───────┘
				 │
				 │(端到端训练二分类模型)
				 ▼
		     ┌────────────────────┐
		     │  results/model.pth │
		     │ (最佳 CNN 权重)    │
		     └────────┬───────────┘
				  │
				  │(加载已训练 CNN)
	    ┌───────────────┴────────────────┐
	    │                                │
	    ▼                                ▼
  ┌───────────────┐                ┌────────────────┐
  │FeatureExtractor│                │   newTrain.py  │
  │  (训练集图像)  │                │ (test 图像)     │
  └───────┬───────┘                └────────┬───────┘
	    │                                  │
	    │(提取 1280 维特征)                │(提取 1280 维特征)
	    ▼                                  ▼
   ┌─────────────┐                     ┌─────────────┐
   │  X_train1280│                     │ X_val1280   │
   └─────┬───────┘                     └────┬────────┘
	   │                                  │
	   │(在训练集上 fit PCA)             │(对验证/测试集做 PCA.transform)
	   ▼                                  ▼
   ┌──────────────┐                  ┌──────────────┐
   │ X_train_pca128│                 │ X_val_pca128 │
   └─────┬────────┘                  └────┬─────────┘
	   │                                  │
	   │(与 y_train 一起训练 LR)         │(用 LR 预测 y_pred)
	   ▼                                  ▼
   ┌────────────────────┐          ┌─────────────────────┐
   │ Logistic Regression│          │  Accuracy on test   │
   │ (传统分类器)       │          │ (PCA+LR 最终表现)   │
   └────────────────────┘          └─────────────────────┘
```

用脚本的时间顺序看：

1. 先跑 train.py：得到深度 CNN 模型的最佳权重 results/model.pth。
2. 再跑 FeatureExtractor.py：用 model.pth 在训练集上提特征 + PCA，生成：
   - results/features_pca128.npy
   - results/labels.npy
   - results/pca_model.pth
3. 最后跑 newTrain.py：
   - 仍然用 model.pth 抽取 dataset/test 的深度特征；
   - 用 pca_model.pth 把这些特征投影到 128 维；
   - 在训练集 PCA 特征上训练 Logistic Regression；
   - 在测试集特征上评估，得到最终准确率。


