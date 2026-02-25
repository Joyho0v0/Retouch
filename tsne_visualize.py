import os
import csv
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from EfficientNet_B0 import EfficientNetB0
from train import get_transforms


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_selected_indices(selector_path):
    if not os.path.exists(selector_path):
        raise FileNotFoundError("selector file not found: " + selector_path)

    with open(selector_path, "rb") as f:
        selector = pickle.load(f)

    if not isinstance(selector, dict):
        raise TypeError("selector must be a dict")

    if "selected_indices" not in selector:
        raise KeyError("selector missing key: selected_indices")

    indices = list(selector["selected_indices"])
    if len(indices) == 0:
        raise ValueError("selected_indices is empty")

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

    def forward_selected_features(self, x):
        features = self.backbone.forward_features(x, pool=True, flatten=True)
        features = features.index_select(dim=1, index=self.selected_indices)
        return features


def load_model_list(summary_csv):
    if not os.path.exists(summary_csv):
        raise FileNotFoundError("summary csv not found: " + summary_csv)

    items = []
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "k" not in row or "save_path" not in row:
                continue
            try:
                k = int(float(row["k"]))
            except ValueError:
                continue
            save_path = row["save_path"].strip()
            if save_path == "":
                continue
            items.append({"k": k, "save_path": save_path})

    return items


def build_test_loader(base_dir, batch_size, num_workers):
    test_dir = os.path.join(base_dir, "dataset/ali", "test")
    _, val_transform = get_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader

def build_train_loader(base_dir, batch_size, num_workers):
    train_dir = os.path.join(base_dir, "dataset/ali", "train")
    _, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader


def collect_features(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.forward_selected_features(images)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def load_weights_for_selected_model(model, state_dict):
    # 兼容两种权重格式：
    # 1) 带有 "backbone." 前缀（来自 EfficientNetB0SelectedChannels）
    # 2) 不带前缀（来自 EfficientNetB0）
    has_backbone_prefix = False
    for key in state_dict.keys():
        if key.startswith("backbone."):
            has_backbone_prefix = True
            break

    if has_backbone_prefix:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.backbone.load_state_dict(state_dict, strict=False)


def subsample(features, labels, max_samples, seed):
    if max_samples is None or max_samples <= 0:
        return features, labels

    n = features.shape[0]
    if n <= max_samples:
        return features, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], labels[idx]


def run_tsne(features, random_state, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = features.shape[0]
    if n < 3:
        raise ValueError("not enough samples for t-SNE")

    pca_dim = features.shape[1]
    if pca_dim > 50:
        pca_dim = 50

    Xp = PCA(n_components=pca_dim, random_state=random_state).fit_transform(features)

    max_perp = (n - 1) / 3.0
    use_perp = perplexity
    if use_perp > max_perp:
        use_perp = max_perp
    if use_perp < 1.0:
        use_perp = 1.0

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=float(use_perp),
        random_state=random_state,
    )
    Z = tsne.fit_transform(Xp)
    return Z


def plot_tsne(Z, labels, title, save_path):
    import matplotlib.pyplot as plt

    classes = np.unique(labels)
    plt.figure(figsize=(6, 5))

    for c in classes:
        mask = labels == c
        x = Z[mask, 0]
        y = Z[mask, 1]
        plt.scatter(x, y, s=8, alpha=0.75, label=str(c))

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", frameon=False, markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    selector_path = os.path.join(project_dir, "nmi_channel_selector.pkl")
    summary_csv = os.path.join(project_dir, "result", "selected_k_train_summary.csv")
    out_dir = os.path.join(project_dir, "result", "tsne")

    batch_size = 16
    num_workers = 2
    random_state = 42
    max_samples = 2000
    perplexity = 30.0

    ensure_dir(out_dir)

    all_indices = load_selected_indices(selector_path)
    model_list = load_model_list(summary_csv)
    if len(model_list) == 0:
        print("no models found in summary csv")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_loader = build_test_loader(project_dir, batch_size, num_workers)
    train_loader = build_train_loader(project_dir, batch_size, num_workers)

    for item in model_list:
        k = item["k"]
        save_path = item["save_path"]
        model_path = save_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_dir, model_path.lstrip("./"))

        if not os.path.exists(model_path):
            print("model not found: " + model_path)
            continue

        selected_indices = all_indices[:k]
        model = EfficientNetB0SelectedChannels(num_classes=2, selected_indices=selected_indices)
        model = model.to(device)

        state_dict = torch.load(model_path, map_location=device)
        load_weights_for_selected_model(model, state_dict)

        # features, labels = collect_features(model, test_loader, device)
        features, labels = collect_features(model, train_loader, device)
        features, labels = subsample(features, labels, max_samples, random_state)

        Z = run_tsne(features, random_state, perplexity)
        title = "t-SNE for k=" + str(k)
        file_name = "tsne_k" + str(k) + ".png"
        save_path = os.path.join(out_dir, file_name)
        plot_tsne(Z, labels, title, save_path)
        print("saved: " + save_path)


if __name__ == "__main__":
    main()
