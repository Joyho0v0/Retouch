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
import pickle

def evaluate_pca_performance():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 1. 加载原始模型P（用于提取验证集特征）
	model = build_model(num_classes=2)
	model.load_state_dict(torch.load("./results/model.pth", map_location=device))
	model = model.to(device)
	model.eval()

	# 2. 加载验证集
	_, val_transform = get_transforms()
	val_dataset = datasets.ImageFolder("./dataset/test", transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

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

	# 4. 加载 PCA 模型，并 transform 验证集特征
	# 说明：
	# - nmi_channel_selector.pkl 是通道选择器（dict），不是 PCA 模型，不能调用 transform。
	# - PCA 模型一般保存在 results/pca_model.pth（通过 joblib.dump 保存）。
	pca_path = "./results/pca_model.pth"
	if not os.path.exists(pca_path):
		raise FileNotFoundError(
			f"找不到 PCA 模型文件: {pca_path}。"
			"请先运行 FeatureExtractor.py 或 FeatureExtractorV1.py 生成 PCA 模型。"
		)

	pca = joblib.load(pca_path)
	if not hasattr(pca, "transform"):
		raise TypeError(
			f"加载的对象不是 PCA 模型（没有 transform 方法）。文件: {pca_path}，对象类型: {type(pca)}"
		)

	X_val_pca = pca.transform(X_val_1280)  # 用训练集学的 PCA 变换验证集
	print(f"Val PCA features shape: {X_val_pca.shape}")

	# 5. 加载训练集 PCA 特征和标签（用于训练分类器）
	X_train_pca = np.load("./results/features_pca128.npy")
	y_train = np.load("./results/labels.npy")

	# 6. 训练简单分类器（Logistic Regression）
	print("Training Logistic Regression on train PCA features...")
	clf = LogisticRegression(max_iter=1000, random_state=42)
	clf.fit(X_train_pca, y_train)

	# 7. 在验证集 PCA 特征上预测
	y_pred = clf.predict(X_val_pca)
	acc = accuracy_score(y_val, y_pred)
	
	print("\n" + "="*50)
	print(f"Final Evaluation on VALIDATION SET")
	print(f"PCA (128-dim) + Logistic Regression Accuracy: {acc:.4f} ({acc*100:.2f}%)")
	print("="*50)


def _load_channel_selector(selector_path, k=None):
	"""加载通道选择器，返回通道索引列表。

	selector_path: pkl 文件路径（通常是 nmi_channel_selector.pkl）
	k: 如果不为 None，则只取前 k 个通道
	"""
	if not os.path.exists(selector_path):
		raise FileNotFoundError(f"找不到通道选择器文件: {selector_path}")

	with open(selector_path, 'rb') as f:
		selector = pickle.load(f)

	if not isinstance(selector, dict):
		raise TypeError(f"通道选择器格式不对，期望 dict，实际: {type(selector)}")

	if 'selected_indices' not in selector:
		raise KeyError("通道选择器缺少 key: selected_indices")

	indices = selector['selected_indices']
	indices = list(indices)

	if k is not None:
		k = int(k)
		indices = indices[:k]

	if len(indices) == 0:
		raise ValueError("通道选择器为空，没有任何 selected_indices")

	return indices


def evaluate_channel_selector_performance():
	"""用通道选择器筛选特征后再训练/评估。

	流程：
	1) 用训练好的 CNN 提取测试集(或验证集) 1280维特征
	2) 用 selector 选出 k 个通道
	3) 用训练集 features_1280.npy 做同样切片，训练 Logistic Regression
	4) 在测试集上评估
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# ========== 参数 ==========
	selector_path = "./nmi_channel_selector.pkl"
	k = 128
	train_features_path = "./results/features_1280.npy"
	train_labels_path = "./results/labels.npy"
	save_filtered_train = True
	save_model = True

	# ========== 加载通道选择器 ==========
	selected_indices = _load_channel_selector(selector_path, k=k)
	print(f"加载通道选择器: {selector_path}")
	print(f"使用前 {len(selected_indices)} 个通道")

	# ========== 加载训练集特征并做通道筛选 ==========
	if not os.path.exists(train_features_path):
		raise FileNotFoundError(
			f"找不到训练集特征文件: {train_features_path}。"
			"请先运行 FeatureExtractor.py 或 ChannelSelect.py 生成 features_1280.npy 和 labels.npy"
		)
	if not os.path.exists(train_labels_path):
		raise FileNotFoundError(
			f"找不到训练集标签文件: {train_labels_path}。"
			"请先运行 FeatureExtractor.py 或 ChannelSelect.py 生成 labels.npy"
		)

	X_train_1280 = np.load(train_features_path)
	y_train = np.load(train_labels_path)

	X_train_sel = X_train_1280[:, selected_indices]
	print(f"Train selected features shape: {X_train_sel.shape}")

	if save_filtered_train:
		save_path = f"./results/features_selected_{len(selected_indices)}.npy"
		np.save(save_path, X_train_sel)
		print(f"已保存筛选后的训练特征: {save_path}")

	# ========== 加载 CNN 模型，用于提取测试集特征 ==========
	model = build_model(num_classes=2)
	model.load_state_dict(torch.load("./results/model.pth", map_location=device))
	model = model.to(device)
	model.eval()

	# ========== 加载测试集 ==========
	_, val_transform = get_transforms()
	val_dataset = datasets.ImageFolder("./dataset/test", transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

	# ========== 提取测试集 1280 维特征 ==========
	print("Extracting test set features...")
	val_features_1280 = []
	val_labels = []
	with torch.no_grad():
		for images, labels in tqdm(val_loader, desc="Test Feature Extraction"):
			images = images.to(device)
			feats = model.forward_features(images, pool=True, flatten=False)
			feats = feats.view(feats.size(0), -1).cpu().numpy()
			val_features_1280.append(feats)
			val_labels.append(labels.numpy())

	X_val_1280 = np.concatenate(val_features_1280, axis=0)
	y_val = np.concatenate(val_labels, axis=0)
	X_val_sel = X_val_1280[:, selected_indices]
	print(f"Test selected features shape: {X_val_sel.shape}")

	# ========== 训练分类器并评估 ==========
	print("Training Logistic Regression on selected features...")
	clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
	clf.fit(X_train_sel, y_train)

	y_pred = clf.predict(X_val_sel)
	acc = accuracy_score(y_val, y_pred)

	print("\n" + "="*50)
	print(f"Final Evaluation on TEST SET")
	print(f"Selected Channels ({len(selected_indices)}-dim) + Logistic Regression Accuracy: {acc:.4f} ({acc*100:.2f}%)")
	print("="*50)

	if save_model:
		model_path = f"./results/logreg_selected_{len(selected_indices)}.joblib"
		joblib.dump(clf, model_path)
		print(f"已保存分类器: {model_path}")

	return acc

if __name__ == "__main__":
	# 你可以二选一运行：
	# 1) PCA 特征 + Logistic Regression
	# evaluate_pca_performance()
	# 2) 通道选择器筛选特征 + Logistic Regression
	evaluate_channel_selector_performance()