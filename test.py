import os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from EfficientNet_B0 import EfficientNetB0
from train_assumpt import EfficientNetB0SelectedChannels


def get_transform():
	test_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return test_transform


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

	test_transform = get_transform()
	test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_path = "./result_assumpt/model_final_k32.pth"
	if not os.path.exists(model_path):
		print("Model not found:")
		print(model_path)
		return

	# 先加载 state_dict，从中读取 selected_indices 来构建模型
	state_dict = torch.load(model_path, map_location=device, weights_only=True)
	selected_indices = state_dict["selected_indices"].tolist()
	model = EfficientNetB0SelectedChannels(
		num_classes=2, selected_indices=selected_indices, dropout=0.0,
	)
	model.load_state_dict(state_dict)
	model = model.to(device)

	acc = evaluate(model, test_loader, device)
	print("Test Accuracy: {:.4f}".format(acc))


if __name__ == "__main__":
	main()
