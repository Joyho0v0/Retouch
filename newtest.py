
import os
import pickle
import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms

from EfficientNet_B0 import EfficientNetB0


def get_transform():
	test_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return test_transform


def load_selected_indices(selector_path, k=None):
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


def evaluate(model, loader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Test", leave=False):
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, preds = torch.max(outputs, 1)
			correct += torch.sum(preds == labels).item()
			total += labels.size(0)
	acc = correct / total
	return acc


def main():
	# ========== 配置 ==========
	# 源域和跨域测试集
	ali_test_dir = "./dataset/ali/test"
	megvii_test_dir = "./dataset/megvii/test"

	selector_path = "./nmi_channel_selector.pkl"
	original_model_path = "./results/OriginalModel.pth"  # 原模型路径
	RUN_MULTI_K = True
	K_LIST = [32, 64, 128, 256]
	selected_k = 128  # RUN_MULTI_K=False 时使用
	num_classes = 2

	save_dir = "./result"
	total_csv = os.path.join(save_dir, "selected_k_total_summary.csv")

	if RUN_MULTI_K:
		k_list = [int(k) for k in K_LIST]
	else:
		k_list = [int(selected_k)]

	# ========== 数据 ==========
	test_transform = get_transform()

	ali_dataset = datasets.ImageFolder(ali_test_dir, transform=test_transform)
	ali_loader = DataLoader(ali_dataset, batch_size=16, shuffle=False, num_workers=2)

	megvii_dataset = datasets.ImageFolder(megvii_test_dir, transform=test_transform)
	megvii_loader = DataLoader(megvii_dataset, batch_size=16, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"device: {device}")
	print(f"k_list: {k_list}")

	# ========== 测试循环 ==========
	with open(total_csv, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["k", "test_acc_ali", "test_acc_megvii", "model_path"],
		)
		writer.writeheader()

		# ---------- 先测试原模型 (k=-1) ----------
		if os.path.exists(original_model_path):
			print("=" * 60)
			print("[0] 测试原模型 (k=-1)")
			original_model = EfficientNetB0(num_classes=num_classes)
			original_model = original_model.to(device)
			state_dict = torch.load(original_model_path, map_location=device)
			original_model.load_state_dict(state_dict)

			print("  源域 (ali):")
			acc_ali_orig = evaluate(original_model, ali_loader, device)
			print("  Ali  Test Accuracy: {:.4f}".format(acc_ali_orig))

			print("  跨域 (megvii):")
			acc_megvii_orig = evaluate(original_model, megvii_loader, device)
			print("  Megvii Test Accuracy: {:.4f}".format(acc_megvii_orig))

			writer.writerow({
				"k": -1,
				"test_acc_ali": float(acc_ali_orig),
				"test_acc_megvii": float(acc_megvii_orig),
				"model_path": original_model_path,
			})
			f.flush()
			del original_model
		else:
			print(f"[Warning] 原模型不存在: {original_model_path}，跳过 k=-1")

		# ---------- 测试各个 k 的通道选择模型 ----------
		for idx_k, k in enumerate(k_list, start=1):
			model_path = os.path.join(save_dir, f"model_selected_{k}.pth")
			if not os.path.exists(model_path):
				print("=" * 60)
				print(f"[{idx_k}/{len(k_list)}] 跳过 k={k} (model not found)")
				print(model_path)
				writer.writerow({
					"k": k,
					"test_acc_ali": "",
					"test_acc_megvii": "",
					"model_path": model_path,
				})
				f.flush()
				continue

			selected_indices = load_selected_indices(selector_path, k=k)
			model = EfficientNetB0SelectedChannels(
				num_classes=num_classes, selected_indices=selected_indices
			)
			model = model.to(device)

			state_dict = torch.load(model_path, map_location=device)
			model.load_state_dict(state_dict)

			print("=" * 60)
			print(f"[{idx_k}/{len(k_list)}] 测试 k={k}")

			# 测试源域 (ali)
			print("  源域 (ali):")
			acc_ali = evaluate(model, ali_loader, device)
			print("  Ali  Test Accuracy: {:.4f}".format(acc_ali))

			# 测试跨域 (megvii)
			print("  跨域 (megvii):")
			acc_megvii = evaluate(model, megvii_loader, device)
			print("  Megvii Test Accuracy: {:.4f}".format(acc_megvii))

			writer.writerow({
				"k": k,
				"test_acc_ali": float(acc_ali),
				"test_acc_megvii": float(acc_megvii),
				"model_path": model_path,
			})
			f.flush()

	print("全部测试完成。Summary saved to:")
	print(total_csv)


if __name__ == "__main__":
	main()

