
import os
import pickle
import csv

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets

from EfficientNet_B0 import EfficientNetB0
from train import get_transforms


def set_seed(seed):
	"""固定随机种子（可选）"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)


def load_selected_indices(selector_path, k=None):
	"""从通道选择器 pkl 里读取 selected_indices。

	selector_path: 例如 ./nmi_channel_selector.pkl
	k: 只取前 k 个通道；None 表示全取
	"""
	if not os.path.exists(selector_path):
		raise FileNotFoundError(f"找不到通道选择器文件: {selector_path}")

	with open(selector_path, "rb") as f:
		selector = pickle.load(f)

	if not isinstance(selector, dict):
		raise TypeError(f"通道选择器格式不对，期望 dict，实际: {type(selector)}")

	if "selected_indices" not in selector:
		raise KeyError("通道选择器缺少 key: selected_indices")

	indices = list(selector["selected_indices"])
	if k is not None:
		k = int(k)
		indices = indices[:k]

	if len(indices) == 0:
		raise ValueError("selected_indices 为空")

	return indices


class EfficientNetB0SelectedChannels(nn.Module):
	"""EfficientNet-B0 + 特征通道筛选 + 新分类头。

	说明：
	- 输入仍然是 RGB 图片
	- 先得到 1280 维全局特征
	- 再按 selected_indices 选出 k 维
	- 最后用线性层分类
	"""

	def __init__(self, num_classes, selected_indices, dropout=0.2):
		super().__init__()
		self.backbone = EfficientNetB0(num_classes=num_classes)
		idx = torch.tensor(selected_indices, dtype=torch.long)
		self.register_buffer("selected_indices", idx)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(int(idx.numel()), num_classes)

	def forward(self, x):
		features = self.backbone.forward_features(x, pool=True, flatten=True)
		features = features.index_select(dim=1, index=self.selected_indices)
		features = self.dropout(features)
		logits = self.fc(features)
		return logits


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


def main():
	# ========== 配置 ==========
	set_seed(42)

	base_dir = "./dataset/ali"
	train_dir = os.path.join(base_dir, "train")
	val_dir = os.path.join(base_dir, "val")

	selector_path = "./nmi_channel_selector.pkl"
	# 你之前的选择是“2/2”：默认用 128，并且希望依次跑多个 k
	RUN_MULTI_K = True
	K_LIST = [32, 64, 128, 256]
	selected_k = 128  # RUN_MULTI_K=False 时使用
	num_classes = 2

	batch_size = 16
	num_workers = 2
	lr = 1e-4
	epochs = 100
	patience = 10

	save_dir = "./result"
	os.makedirs(save_dir, exist_ok=True)
	summary_csv = os.path.join(save_dir, "selected_k_train_summary.csv")

	# ========== 通道选择器 ==========
	print(f"加载通道选择器: {selector_path}")
	all_indices = load_selected_indices(selector_path, k=None)
	print(f"选择器总通道数: {len(all_indices)}")

	if RUN_MULTI_K:
		k_list = [int(k) for k in K_LIST]
	else:
		k_list = [int(selected_k)]

	if any(k <= 0 for k in k_list):
		raise ValueError(f"K_LIST 里存在非正数: {k_list}")
	max_k = max(k_list)
	if max_k > len(all_indices):
		raise ValueError(
			f"最大 k={max_k} 超过选择器通道数={len(all_indices)}，请调小 K_LIST"
		)

	# ========== 数据 ==========
	train_transform, val_transform = get_transforms()
	train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
	val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	# ========== 设备 / 损失 ==========
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	criterion = nn.CrossEntropyLoss()

	print("开始训练...")
	print(f"device: {device}")
	print(f"k_list: {k_list}")

	with open(summary_csv, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["k", "best_val_acc", "epochs_ran", "save_path"],
		)
		writer.writeheader()

		for idx_k, k in enumerate(k_list, start=1):
			save_path = os.path.join(save_dir, f"model_selected_{k}.pth")
			selected_indices = all_indices[:k]

			print("=" * 60)
			print(f"[{idx_k}/{len(k_list)}] 训练 k={k}")
			print(f"save_path: {save_path}")

			model = EfficientNetB0SelectedChannels(num_classes=num_classes, selected_indices=selected_indices)
			model = model.to(device)
			optimizer = optim.Adam(model.parameters(), lr=lr)

			best_acc = 0.0
			no_improve_epochs = 0
			epochs_ran = 0

			for epoch in range(1, epochs + 1):
				epochs_ran = epoch
				train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
				val_loss, val_acc = evaluate(model, val_loader, criterion, device)

				is_best = val_acc > best_acc
				if is_best:
					best_acc = val_acc
					torch.save(model.state_dict(), save_path)
					no_improve_epochs = 0
				else:
					no_improve_epochs += 1

				print("Epoch {}/{}".format(epoch, epochs))
				print("Train Loss: {:.4f}, Train Acc: {:.4f}".format(train_loss, train_acc))
				print("Val   Loss: {:.4f}, Val   Acc: {:.4f}".format(val_loss, val_acc))
				print("Best  Acc : {:.4f}".format(best_acc))
				print("No improve epochs: {} / {}".format(no_improve_epochs, patience))
				print("-" * 50)

				if no_improve_epochs >= patience:
					print("Validation accuracy did not improve. Early stopping.")
					break

			writer.writerow(
				{
					"k": k,
					"best_val_acc": float(best_acc),
					"epochs_ran": int(epochs_ran),
					"save_path": save_path,
				}
			)
			f.flush()

	print("全部训练完成。Summary saved to:")
	print(summary_csv)


if __name__ == "__main__":
	main()

