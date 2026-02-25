"""Readable EfficientNet-B0 re-implementation without fancy syntax."""

import torch
from torch import nn


def drop_connect(x, drop_prob, training):
	keep_prob = 1.0 - drop_prob
	if not training or drop_prob == 0.0:
		return x
	random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
	random_tensor = random_tensor.floor()
	return x * random_tensor / keep_prob


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
		if self.use_residual:
			out = drop_connect(out, self.drop_connect_rate, self.training)
			out = out + x
		return out



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

	def forward_features(self, x, pool=True, flatten=True):
		x = self.stem(x)
		x = self.blocks(x)
		x = self.head(x)
		if pool:
			x = self.avgpool(x)
		if flatten:
			x = torch.flatten(x, 1)
		return x

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

	def forward(self, x):
		x = self.forward_features(x, pool=True, flatten=True)
		x = self.classifier(x)
		return x

	def create_feature_extractor(self, pool=True, flatten=True):
		return FeatureExtractor(self, pool=pool, flatten=flatten)


class FeatureExtractor(nn.Module):
	def __init__(self, backbone, pool=True, flatten=True):
		super().__init__()
		self.backbone = backbone
		self.pool = pool
		self.flatten = flatten

	def forward(self, x):
		return self.backbone.forward_features(x, pool=self.pool, flatten=self.flatten)


def efficientnet_b0(num_classes=1000, drop_connect_rate=0.2):
	return EfficientNetB0(num_classes=num_classes, drop_connect_rate=drop_connect_rate)


__all__ = ["EfficientNetB0", "efficientnet_b0", "FeatureExtractor"]
