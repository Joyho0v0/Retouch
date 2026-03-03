"""
数据分阶段训练方案（猜想二）

！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

等后面数据集足够大时再完善此方案，目前仅做框架搭建和流程验证。

！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


整体流程：
  Step 1: 将 ali/train 按比例分成 Part A 和 Part B（分层采样，类别平衡）
  Step 2: 用 Part A 训练 OriginalModel（基模型）
  Step 3: 用 Part A + OriginalModel 提取特征 → 贪心通道选择 → 保存选择器
  Step 4: Phase 1 — 冻结 backbone，用 Part A 训练 FC head（让 FC 适配选中通道）
  Step 5: Phase 2 — 解冻 backbone，用 Part B（新鲜数据）微调整个模型

注意：
  此方案需要足够大的数据集才能有效（建议 ali/train >= 5000 张）
  当前 ali/train 只有 1400 张，分割后两边都会数据不足
  验证集 ali/val 始终使用完整数据，不参与分割
"""

import os
import csv
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from EfficientNet_B0 import EfficientNetB0, FeatureExtractor



# 数据目录
BASE_DIR = "./dataset/ali"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# 输出目录（与 newtrain.py 分开，避免覆盖）
SAVE_DIR = "./result_assumpt"

# 数据分割比例
PART_A_RATIO = 0.7  # Part A 占训练集的比例（用于基模型 + 通道选择 + Phase 1）

# 基模型训练参数
BASE_LR = 1e-4
BASE_EPOCHS = 100
BASE_PATIENCE = 10

# 通道选择参数
MAX_K = 256
SELECTED_K = 32              # 最终使用的通道数
SCORE_N_SAMPLES = 3000
CHECKPOINTS = [32, 64, 128, 256]

# Phase 1 参数（冻结 backbone，在 Part A 上训练 FC）
P1_FC_LR = 1e-3
P1_WEIGHT_DECAY = 1e-4
P1_EPOCHS = 30
P1_PATIENCE = 10

# Phase 2 参数（解冻 backbone，在 Part B 上微调）
P2_BACKBONE_LR = 1e-6
P2_FC_LR = 1e-4
P2_WEIGHT_DECAY = 1e-4
P2_LABEL_SMOOTHING = 0.1
P2_EPOCHS = 100
P2_PATIENCE = 15
P2_DROPOUT = 0.4

NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 2
SEED = 42



def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)


def get_train_transform():
	"""基模型训练和 Phase 1 使用的增强（较弱）"""
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])


def get_strong_transform():
	"""Phase 2 使用的强增强"""
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(15),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
		transforms.RandomGrayscale(p=0.1),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
	])


def get_val_transform():
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])


def split_dataset_indices(dataset, ratio, seed=42):
	"""分层采样：按类别比例将数据集索引分成两部分。

	返回: (part_a_indices, part_b_indices)
	"""
	targets = np.array(dataset.targets)
	classes = np.unique(targets)

	rng = np.random.default_rng(seed)

	part_a = []
	part_b = []

	for c in classes:
		c_indices = np.flatnonzero(targets == c)
		rng.shuffle(c_indices)
		split_point = int(len(c_indices) * ratio)
		part_a.append(c_indices[:split_point])
		part_b.append(c_indices[split_point:])

	part_a = np.concatenate(part_a)
	part_b = np.concatenate(part_b)
	rng.shuffle(part_a)
	rng.shuffle(part_b)

	return part_a.tolist(), part_b.tolist()


# 模型定义

class EfficientNetB0SelectedChannels(nn.Module):
	"""EfficientNet-B0 + 通道筛选 + FC head"""

	def __init__(self, num_classes, selected_indices, dropout=0.3):
		super().__init__()
		self.backbone = EfficientNetB0(num_classes=num_classes)
		idx = torch.tensor(selected_indices, dtype=torch.long)
		self.register_buffer("selected_indices", idx)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(int(idx.numel()), num_classes)

	def load_pretrained_backbone(self, pretrained_path, device="cpu"):
		if not os.path.exists(pretrained_path):
			print(f"[Warning] 预训练模型不存在: {pretrained_path}")
			return
		state_dict = torch.load(pretrained_path, map_location=device)
		self.backbone.load_state_dict(state_dict)
		print(f"已加载预训练 backbone: {pretrained_path}")

	def freeze_backbone(self):
		for param in self.backbone.parameters():
			param.requires_grad = False
		print("Backbone 已冻结")

	def unfreeze_backbone(self):
		for param in self.backbone.parameters():
			param.requires_grad = True
		print("Backbone 已解冻")

	def forward(self, x):
		features = self.backbone.forward_features(x, pool=True, flatten=True)
		features = features.index_select(dim=1, index=self.selected_indices)
		features = self.dropout(features)
		logits = self.fc(features)
		return logits


# 训练 / 评估 函数

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

	return running_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):
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

	return running_loss / total, correct / total


# Step 1: 分割数据

def step1_split_data():
	"""分割 ali/train 为 Part A 和 Part B"""
	print("=" * 60)
	print("Step 1: 分割训练数据")
	print("=" * 60)

	# 先用一个临时 dataset 来获取 targets（不需要 transform）
	temp_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_val_transform())
	part_a_idx, part_b_idx = split_dataset_indices(temp_dataset, PART_A_RATIO, SEED)

	print(f"总训练样本: {len(temp_dataset)}")
	print(f"Part A (基模型 + 通道选择 + Phase 1): {len(part_a_idx)} 张")
	print(f"Part B (Phase 2 微调): {len(part_b_idx)} 张")

	# 检查类别平衡
	targets = np.array(temp_dataset.targets)
	for c in np.unique(targets):
		a_count = np.sum(targets[part_a_idx] == c)
		b_count = np.sum(targets[part_b_idx] == c)
		print(f"  类别 {c}: Part A={a_count}, Part B={b_count}")

	return part_a_idx, part_b_idx


# Step 2: 用 Part A 训练基模型

def step2_train_base_model(part_a_idx, device):
	"""用 Part A 数据训练 OriginalModel"""
	print("\n" + "=" * 60)
	print("Step 2: 用 Part A 训练基模型 (OriginalModel)")
	print("=" * 60)

	save_path = os.path.join(SAVE_DIR, "OriginalModel_partA.pth")

	# 创建 Part A 数据集
	train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
	part_a_dataset = Subset(train_dataset, part_a_idx)
	part_a_loader = DataLoader(
		part_a_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
	)

	# 验证集始终用完整 val
	val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
	val_loader = DataLoader(
		val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
	)

	model = EfficientNetB0(num_classes=NUM_CLASSES)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=BASE_LR)

	best_acc = 0.0
	no_improve = 0
	epoch_csv = os.path.join(SAVE_DIR, "step2_base_model_log.csv")

	with open(epoch_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc"])
		writer.writeheader()

		for epoch in range(1, BASE_EPOCHS + 1):
			train_loss, train_acc = train_one_epoch(model, part_a_loader, criterion, optimizer, device)
			val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

			if val_acc > best_acc:
				best_acc = val_acc
				torch.save(model.state_dict(), save_path)
				no_improve = 0
			else:
				no_improve += 1

			writer.writerow({
				"epoch": epoch,
				"train_loss": round(train_loss, 6),
				"train_acc": round(train_acc, 6),
				"val_loss": round(val_loss, 6),
				"val_acc": round(val_acc, 6),
				"best_acc": round(best_acc, 6),
			})
			f.flush()

			print("Epoch {}/{} | Train Acc: {:.4f} | Val Acc: {:.4f} | Best: {:.4f} | No improve: {}/{}".format(
				epoch, BASE_EPOCHS, train_acc, val_acc, best_acc, no_improve, BASE_PATIENCE
			))

			if no_improve >= BASE_PATIENCE:
				print("Early stopping.")
				break

	print(f"基模型训练完成。Best val acc: {best_acc:.4f}")
	print(f"保存到: {save_path}")
	return save_path


# Step 3: 用 Part A + 基模型提取特征，贪心通道选择

def step3_channel_select(part_a_idx, base_model_path, device):
	"""基于 Part A 提取特征并进行通道选择"""
	print("\n" + "=" * 60)
	print("Step 3: 提取特征 + 贪心通道选择")
	print("=" * 60)

	# 加载基模型
	model = EfficientNetB0(num_classes=NUM_CLASSES)
	model.load_state_dict(torch.load(base_model_path, map_location=device))
	model = model.to(device)
	model.eval()

	# 特征提取器
	extractor = FeatureExtractor(model, pool=True, flatten=False)

	# Part A 数据（不做增强）
	dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_val_transform())
	part_a_dataset = Subset(dataset, part_a_idx)
	loader = DataLoader(
		part_a_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
	)

	# 提取特征
	all_features = []
	all_labels = []

	print("提取 Part A 特征...")
	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Feature Extraction"):
			images = images.to(device)
			feats = extractor(images)
			feats = feats.view(feats.size(0), -1).cpu().numpy()
			all_features.append(feats)
			all_labels.append(labels.numpy())

	features = np.concatenate(all_features, axis=0)
	labels = np.concatenate(all_labels, axis=0)
	print(f"特征矩阵 shape: {features.shape}")

	# 保存特征（可选）
	np.save(os.path.join(SAVE_DIR, "features_partA_1280.npy"), features)
	np.save(os.path.join(SAVE_DIR, "labels_partA.npy"), labels)

	# 贪心通道选择（导入 ChannelSelect 中的函数）
	from ChannelSelect import greedy_select_channels_parallel, plot_nmi_curve

	selected_indices, nmi_curve, checkpoint_results = greedy_select_channels_parallel(
		features=features,
		labels=labels,
		max_k=MAX_K,
		n_clusters=None,
		random_state=SEED,
		score_n_samples=SCORE_N_SAMPLES,
		checkpoints=CHECKPOINTS,
	)

	# 画 NMI 曲线
	plot_nmi_curve(
		nmi_curve=nmi_curve,
		checkpoints=CHECKPOINTS,
		save_path=os.path.join(SAVE_DIR, "nmi_curve.png"),
	)

	# 保存选择器
	selector_data = {
		"selected_indices": selected_indices,
		"nmi_curve": nmi_curve,
		"checkpoint_results": checkpoint_results,
		"max_k": MAX_K,
		"strategy": "greedy_kmeans_nmi",
		"data_split": "part_a_only",
	}
	selector_path = os.path.join(SAVE_DIR, "nmi_channel_selector.pkl")
	with open(selector_path, "wb") as f:
		pickle.dump(selector_data, f)
	print(f"通道选择器保存到: {selector_path}")

	# 打印 checkpoint 结果
	for k in CHECKPOINTS:
		if k in checkpoint_results:
			print(f"  k={k}: NMI={checkpoint_results[k]:.6f}")

	return selector_path, selected_indices


# Step 4: Phase 1 — 冻结 backbone，用 Part A 训练 FC head

def step4_phase1(part_a_idx, base_model_path, selected_indices, device):
	"""Phase 1: 冻结 backbone，用 Part A 训练 FC head"""
	print("\n" + "=" * 60)
	print("Step 4: Phase 1 — 冻结 backbone，在 Part A 上训练 FC")
	print("=" * 60)

	k = SELECTED_K
	indices = selected_indices[:k]
	print(f"使用通道数: {k}")

	save_path = os.path.join(SAVE_DIR, f"model_phase1_k{k}.pth")

	# 数据
	train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
	part_a_dataset = Subset(train_dataset, part_a_idx)
	part_a_loader = DataLoader(
		part_a_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
	)

	val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
	val_loader = DataLoader(
		val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
	)

	# 模型
	model = EfficientNetB0SelectedChannels(
		num_classes=NUM_CLASSES, selected_indices=indices, dropout=P2_DROPOUT,
	)
	model.load_pretrained_backbone(base_model_path, device=device)
	model = model.to(device)
	model.freeze_backbone()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.fc.parameters(), lr=P1_FC_LR, weight_decay=P1_WEIGHT_DECAY)

	best_acc = 0.0
	no_improve = 0
	epoch_csv = os.path.join(SAVE_DIR, f"step4_phase1_k{k}_log.csv")

	with open(epoch_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc"])
		writer.writeheader()

		for epoch in range(1, P1_EPOCHS + 1):
			train_loss, train_acc = train_one_epoch(model, part_a_loader, criterion, optimizer, device)
			val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

			if val_acc > best_acc:
				best_acc = val_acc
				torch.save(model.state_dict(), save_path)
				no_improve = 0
			else:
				no_improve += 1

			writer.writerow({
				"epoch": epoch,
				"train_loss": round(train_loss, 6),
				"train_acc": round(train_acc, 6),
				"val_loss": round(val_loss, 6),
				"val_acc": round(val_acc, 6),
				"best_acc": round(best_acc, 6),
			})
			f.flush()

			print("Epoch {}/{} [Phase1] | Train Acc: {:.4f} | Val Acc: {:.4f} | Best: {:.4f} | No improve: {}/{}".format(
				epoch, P1_EPOCHS, train_acc, val_acc, best_acc, no_improve, P1_PATIENCE
			))

			if no_improve >= P1_PATIENCE:
				print("Phase 1 early stopping.")
				break

	print(f"Phase 1 完成。Best val acc: {best_acc:.4f}")
	print(f"保存到: {save_path}")
	return save_path, best_acc


# Step 5: Phase 2 — 解冻 backbone，用 Part B 微调

def step5_phase2(part_b_idx, phase1_model_path, selected_indices, device):
	"""Phase 2: 解冻 backbone，用 Part B（新鲜数据）微调"""
	print("\n" + "=" * 60)
	print("Step 5: Phase 2 — 解冻 backbone，在 Part B 上微调")
	print("=" * 60)

	k = SELECTED_K
	indices = selected_indices[:k]
	print(f"使用通道数: {k}")
	print(f"Part B 样本数: {len(part_b_idx)}")

	save_path = os.path.join(SAVE_DIR, f"model_final_k{k}.pth")

	# 数据：Part B 使用强增强
	train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_strong_transform())
	part_b_dataset = Subset(train_dataset, part_b_idx)
	part_b_loader = DataLoader(
		part_b_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
	)

	val_dataset = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
	val_loader = DataLoader(
		val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
	)

	# 模型：从 Phase 1 的权重开始
	model = EfficientNetB0SelectedChannels(
		num_classes=NUM_CLASSES, selected_indices=indices, dropout=P2_DROPOUT,
	)
	model = model.to(device)
	state_dict = torch.load(phase1_model_path, map_location=device)
	model.load_state_dict(state_dict)
	print(f"已加载 Phase 1 权重: {phase1_model_path}")

	# 解冻 backbone
	model.unfreeze_backbone()

	# 差异化学习率
	backbone_params = list(model.backbone.parameters())
	fc_params = list(model.fc.parameters())

	optimizer = optim.Adam(
		[
			{"params": backbone_params, "lr": P2_BACKBONE_LR},
			{"params": fc_params, "lr": P2_FC_LR},
		],
		weight_decay=P2_WEIGHT_DECAY,
	)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="max", factor=0.5, patience=5,
	)

	criterion = nn.CrossEntropyLoss(label_smoothing=P2_LABEL_SMOOTHING)

	best_acc = 0.0
	no_improve = 0
	epoch_csv = os.path.join(SAVE_DIR, f"step5_phase2_k{k}_log.csv")

	with open(epoch_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc"])
		writer.writeheader()

		for epoch in range(1, P2_EPOCHS + 1):
			train_loss, train_acc = train_one_epoch(model, part_b_loader, criterion, optimizer, device)
			val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
			scheduler.step(val_acc)

			if val_acc > best_acc:
				best_acc = val_acc
				torch.save(model.state_dict(), save_path)
				no_improve = 0
			else:
				no_improve += 1

			writer.writerow({
				"epoch": epoch,
				"train_loss": round(train_loss, 6),
				"train_acc": round(train_acc, 6),
				"val_loss": round(val_loss, 6),
				"val_acc": round(val_acc, 6),
				"best_acc": round(best_acc, 6),
			})
			f.flush()

			print("Epoch {}/{} [Phase2] | Train Acc: {:.4f} | Val Acc: {:.4f} | Best: {:.4f} | No improve: {}/{}".format(
				epoch, P2_EPOCHS, train_acc, val_acc, best_acc, no_improve, P2_PATIENCE
			))

			if no_improve >= P2_PATIENCE:
				print("Phase 2 early stopping.")
				break

	print(f"Phase 2 完成。Best val acc: {best_acc:.4f}")
	print(f"最终模型保存到: {save_path}")
	return save_path, best_acc


# 主函数

def main():
	set_seed(SEED)
	os.makedirs(SAVE_DIR, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# 数据量检查
	temp_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_val_transform())
	total = len(temp_dataset)
	part_a_count = int(total * PART_A_RATIO)
	part_b_count = total - part_a_count

	if total < 3000:
		print("")
		print("!" * 60)
		print(f"[Warning] 训练集仅 {total} 张，分割后 Part A={part_a_count}, Part B={part_b_count}")
		print("数据量偏少，此方案效果可能不佳。建议数据集 >= 5000 张时使用。")
		print("!" * 60)
		print("")

	# Step 1: 分割数据
	part_a_idx, part_b_idx = step1_split_data()

	# Step 2: 用 Part A 训练基模型
	base_model_path = step2_train_base_model(part_a_idx, device)

	# Step 3: 用 Part A 提取特征 + 通道选择
	selector_path, selected_indices = step3_channel_select(part_a_idx, base_model_path, device)

	# Step 4: Phase 1 — 冻结 backbone，Part A 训练 FC head
	phase1_path, phase1_acc = step4_phase1(part_a_idx, base_model_path, selected_indices, device)

	# Step 5: Phase 2 — 解冻 backbone，Part B 微调
	final_path, final_acc = step5_phase2(part_b_idx, phase1_path, selected_indices, device)

	# 汇总
	print("\n" + "=" * 60)
	print("全流程完成！")
	print("=" * 60)
	print(f"数据分割: Part A={len(part_a_idx)}, Part B={len(part_b_idx)}")
	print(f"基模型: {base_model_path}")
	print(f"通道选择器: {selector_path}")
	print(f"Phase 1 最佳 val acc: {phase1_acc:.4f}")
	print(f"Phase 2 最佳 val acc: {final_acc:.4f}")
	print(f"最终模型: {final_path}")

	# 保存汇总信息
	summary_path = os.path.join(SAVE_DIR, "pipeline_summary.csv")
	with open(summary_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=[
			"step", "detail", "value"
		])
		writer.writeheader()
		writer.writerow({"step": "data_split", "detail": "part_a_count", "value": len(part_a_idx)})
		writer.writerow({"step": "data_split", "detail": "part_b_count", "value": len(part_b_idx)})
		writer.writerow({"step": "phase1", "detail": "best_val_acc", "value": round(phase1_acc, 6)})
		writer.writerow({"step": "phase2", "detail": "best_val_acc", "value": round(final_acc, 6)})
		writer.writerow({"step": "output", "detail": "final_model", "value": final_path})
		writer.writerow({"step": "output", "detail": "selector", "value": selector_path})
	print(f"汇总信息: {summary_path}")


if __name__ == "__main__":
	main()
