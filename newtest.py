
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
	base_dir = "./dataset/megvii"
	test_dir = os.path.join(base_dir, "test")

	selector_path = "./nmi_channel_selector.pkl"
	RUN_MULTI_K = True
	K_LIST = [32, 64, 128, 256]
	selected_k = 128  # RUN_MULTI_K=False 时使用
	num_classes = 2
	save_dir = "./result"
	summary_csv = os.path.join(save_dir, "selected_k_test_summary.csv")

	if RUN_MULTI_K:
		k_list = [int(k) for k in K_LIST]
	else:
		k_list = [int(selected_k)]

	test_transform = get_transform()
	test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"device: {device}")
	print(f"k_list: {k_list}")

	with open(summary_csv, "w", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["k", "test_acc", "model_path"],
		)
		writer.writeheader()

		for idx_k, k in enumerate(k_list, start=1):
			model_path = os.path.join(save_dir, f"model_selected_{k}.pth")
			if not os.path.exists(model_path):
				print("=" * 60)
				print(f"[{idx_k}/{len(k_list)}] 跳过 k={k} (model not found)")
				print(model_path)
				writer.writerow({"k": k, "test_acc": "", "model_path": model_path})
				f.flush()
				continue

			selected_indices = load_selected_indices(selector_path, k=k)
			model = EfficientNetB0SelectedChannels(num_classes=num_classes, selected_indices=selected_indices)
			model = model.to(device)

			state_dict = torch.load(model_path, map_location=device)
			model.load_state_dict(state_dict)

			print("=" * 60)
			print(f"[{idx_k}/{len(k_list)}] 测试 k={k}")
			acc = evaluate(model, test_loader, device)
			print("Test Accuracy: {:.4f}".format(acc))

			writer.writerow({"k": k, "test_acc": float(acc), "model_path": model_path})
			f.flush()

	print("全部测试完成。Summary saved to:")
	print(summary_csv)


if __name__ == "__main__":
	main()

