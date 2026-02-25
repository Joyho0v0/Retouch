import os
import csv
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
from torchvision import datasets, transforms
from EfficientNet_B0 import EfficientNetB0
import numpy as np


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


def build_model(num_classes):
	# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
	# in_features = model.classifier[1].in_features
	# model.classifier[1] = nn.Linear(in_features, num_classes)
	# return model
	return EfficientNetB0(num_classes=num_classes)

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
	base_dir = "./dataset/ali"
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
	save_path = "./results/OriginalModel.pth"

	# 为每个 epoch 记录原始模型的 train_acc 和 val_acc
	original_csv = "./results/original.csv"
	os.makedirs(os.path.dirname(original_csv), exist_ok=True)

	with open(original_csv, "w", newline="") as f_csv:
		writer = csv.DictWriter(
			f_csv,
			fieldnames=["k", "epoch", "train_acc", "val_acc"],
		)
		writer.writeheader()

		for epoch in range(1, epochs + 1):
			train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
			val_loss, val_acc = evaluate(model, val_loader, criterion, device)

			# 记录到 original.csv，这里 k 固定为 -1
			writer.writerow(
				{
					"k": -1,
					"epoch": int(epoch),
					"train_acc": float(train_acc),
					"val_acc": float(val_acc),
				}
			)
			f_csv.flush()

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


if __name__ == "__main__":
	main()
