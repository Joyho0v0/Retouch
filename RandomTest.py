"""
随机通道选择器精度测试：
  对比原始模型（端到端）与使用 random_channel_selector.pkl 选出的通道 + 逻辑回归的精度。
  在 ali/test 和 megvii/test 上分别测试。

使用方法:
    conda activate Retouch
    python RandomTest.py

代码风格：不使用 lambda、argparse 等复杂句式，保持清晰简洁。
"""

import os
import csv
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from EfficientNet_B0 import EfficientNetB0, FeatureExtractor


# ========== 配置 ==========
MODEL_PATH = "./results/OriginalModel.pth"
PKL_PATH = "./random_channel_selector.pkl"
ALI_TRAIN_DIR = "./dataset/ali/train"
ALI_TEST_DIR = "./dataset/ali/test"
MEGVII_TEST_DIR = "./dataset/megvii/test"
CSV_SAVE_PATH = "./results/originVSselect.csv"
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 2


# ========== 数据预处理 ==========
def get_val_transform():
    """验证/测试用的图像预处理。"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform


# ========== 端到端精度测试 ==========
def test_model_accuracy(model, data_dir, name, device):
    """
    用原始模型做端到端推理，返回分类准确率。

    输入:
        model: 已加载好的 EfficientNetB0 模型
        data_dir: 测试集目录
        name: 数据集描述
        device: 计算设备
    返回:
        acc: 准确率
    """
    dataset = datasets.ImageFolder(data_dir, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{name} Inference"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct = correct + torch.sum(preds == labels).item()
            total = total + labels.size(0)

    acc = correct / total
    return acc


# ========== 特征提取 ==========
def extract_features(data_dir, name, model, device):
    """
    用模型提取 1280 维特征（去掉分类头，只保留特征提取 + 全局平均池化）。

    输入:
        data_dir: 数据集目录
        name: 数据集描述
        model: 已加载好的 EfficientNetB0 模型
        device: 计算设备
    返回:
        X: 特征矩阵 [N, 1280]
        y: 标签数组 [N,]
    """
    extractor = FeatureExtractor(model, pool=True, flatten=False)

    dataset = datasets.ImageFolder(data_dir, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_features = []
    all_labels = []

    print(f"提取 {name} 特征...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{name} Feature Extraction"):
            images = images.to(device)
            feats = extractor(images)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"  {name} 特征 shape: {X.shape}")
    return X, y


# ========== 逻辑回归训练与测试 ==========
def train_and_test_lr(train_X, train_y, test_X, test_y, name):
    """
    在训练特征上拟合逻辑回归，在测试特征上计算准确率。

    输入:
        train_X: 训练特征 [N_train, dim]
        train_y: 训练标签 [N_train,]
        test_X: 测试特征 [N_test, dim]
        test_y: 测试标签 [N_test,]
        name: 描述文字
    返回:
        acc: 测试准确率
    """
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    lr = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
    lr.fit(train_X_scaled, train_y)

    preds = lr.predict(test_X_scaled)
    correct = int(np.sum(preds == test_y))
    total = len(test_y)
    acc = correct / total

    print(f"  {name}: {correct}/{total} = {acc:.4f}")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==================== 加载原始模型 ====================
    print("")
    print("=" * 60)
    print("加载原始模型")
    print("=" * 60)

    model = EfficientNetB0(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"已加载: {MODEL_PATH}")

    # ==================== 加载随机通道选择器 ====================
    print("")
    print("=" * 60)
    print("加载随机通道选择器")
    print("=" * 60)

    if not os.path.exists(PKL_PATH):
        print(f"文件不存在: {PKL_PATH}")
        print("请先运行 RandomSelect.py 生成通道选择器")
        return

    with open(PKL_PATH, "rb") as f:
        selector = pickle.load(f)

    selected_indices = selector["selected_indices"]
    best_nmi = selector.get("best_nmi_ali_test", "未知")
    total_loops = selector.get("total_loops", "未知")
    select_k = len(selected_indices)
    print(f"选中通道数: {select_k}")
    print(f"最佳 NMI (ali/test): {best_nmi}")
    print(f"搜索轮数: {total_loops}")

    # ==================== 测试 1: 原始模型端到端精度 ====================
    print("")
    print("=" * 60)
    print("测试 1: 原始模型端到端精度 (EfficientNetB0, Dropout+Linear)")
    print("=" * 60)

    acc_ali_e2e = test_model_accuracy(model, ALI_TEST_DIR, "ali/test", device)
    print(f"  ali/test 精度: {acc_ali_e2e:.4f}")

    acc_meg_e2e = test_model_accuracy(model, MEGVII_TEST_DIR, "megvii/test", device)
    print(f"  megvii/test 精度: {acc_meg_e2e:.4f}")

    # ==================== 提取各数据集 1280 维特征 ====================
    print("")
    print("=" * 60)
    print("提取各数据集 1280 维特征")
    print("=" * 60)

    train_X, train_y = extract_features(ALI_TRAIN_DIR, "ali/train", model, device)
    ali_test_X, ali_test_y = extract_features(ALI_TEST_DIR, "ali/test", model, device)
    meg_test_X, meg_test_y = extract_features(MEGVII_TEST_DIR, "megvii/test", model, device)

    # ==================== 测试 2: 全 1280 通道 + 逻辑回归 ====================
    print("")
    print("=" * 60)
    print("测试 2: 全 1280 通道 + 逻辑回归")
    print("=" * 60)

    acc_ali_full = train_and_test_lr(train_X, train_y, ali_test_X, ali_test_y, "ali/test")
    acc_meg_full = train_and_test_lr(train_X, train_y, meg_test_X, meg_test_y, "megvii/test")

    # ==================== 测试 3: 选中通道 + 逻辑回归 ====================
    print("")
    print("=" * 60)
    print(f"测试 3: 选中 {select_k} 通道 + 逻辑回归")
    print("=" * 60)

    train_X_sel = train_X[:, selected_indices]
    ali_test_X_sel = ali_test_X[:, selected_indices]
    meg_test_X_sel = meg_test_X[:, selected_indices]

    acc_ali_sel = train_and_test_lr(train_X_sel, train_y, ali_test_X_sel, ali_test_y, "ali/test")
    acc_meg_sel = train_and_test_lr(train_X_sel, train_y, meg_test_X_sel, meg_test_y, "megvii/test")

    # ==================== 精度对比汇总 ====================
    print("")
    print("=" * 60)
    print("精度对比汇总")
    print("=" * 60)
    print("")
    header = f"  {'方法':<40} {'ali/test':>10} {'megvii/test':>12}"
    print(header)
    print("  " + "-" * 62)
    print(f"  {'原模型 端到端 (Dropout+Linear))':<40} {acc_ali_e2e:>10.4f} {acc_meg_e2e:>12.4f}")
    print(f"  {'全1280通道 + 逻辑回归':<40} {acc_ali_full:>10.4f} {acc_meg_full:>12.4f}")
    print(f"  {'选中{0}通道 + 逻辑回归 (随机搜索)'.format(select_k):<40} {acc_ali_sel:>10.4f} {acc_meg_sel:>12.4f}")
    print("  " + "-" * 62)
    print("")

    # 精度变化分析
    print("精度变化分析:")
    delta_ali = acc_ali_sel - acc_ali_e2e
    delta_meg = acc_meg_sel - acc_meg_e2e
    sign_ali = "+" if delta_ali >= 0 else ""
    sign_meg = "+" if delta_meg >= 0 else ""
    print(f"  ali/test:    通道选择+LR vs 原模型 = {sign_ali}{delta_ali:.4f}")
    print(f"  megvii/test: 通道选择+LR vs 原模型 = {sign_meg}{delta_meg:.4f}")

    delta_ali_full = acc_ali_sel - acc_ali_full
    delta_meg_full = acc_meg_sel - acc_meg_full
    sign_ali_f = "+" if delta_ali_full >= 0 else ""
    sign_meg_f = "+" if delta_meg_full >= 0 else ""
    print(f"  ali/test:    {select_k}通道 vs 全1280通道 (均LR) = {sign_ali_f}{delta_ali_full:.4f}")
    print(f"  megvii/test: {select_k}通道 vs 全1280通道 (均LR) = {sign_meg_f}{delta_meg_full:.4f}")

    # ==================== 保存为 CSV ====================
    csv_dir = os.path.dirname(CSV_SAVE_PATH)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    with open(CSV_SAVE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", "ali/test", "megvii/test"])
        writer.writerow(["原模型 端到端 (Dropout+Linear)", f"{acc_ali_e2e:.4f}", f"{acc_meg_e2e:.4f}"])
        writer.writerow(["全1280通道 + 逻辑回归", f"{acc_ali_full:.4f}", f"{acc_meg_full:.4f}"])
        writer.writerow([f"选中{select_k}通道 + 逻辑回归 (随机搜索)", f"{acc_ali_sel:.4f}", f"{acc_meg_sel:.4f}"])

    print("")
    print(f"结果已保存到: {CSV_SAVE_PATH}")


if __name__ == "__main__":
    main()
