from train import *
from EfficientNet_B0 import FeatureExtractor as EfficientNetFeatureExtractor


# 特征提取+PCA降维
def extract_and_reduce_features():
	base_dir = "./dataset/ali"
	train_dir = os.path.join(base_dir,"train")

	# 使用验证转换（无数据增强）
	_, val_transform = get_transforms()
	train_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
	train_loader = DataLoader(train_dataset,batch_size=16,shuffle=False,num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 加载训练好的模型
	model = build_model(num_classes=2)
	model.load_state_dict(torch.load("./results/OriginalModel.pth",map_location=device))
	model = model.to(device)
	model.eval()
	
	# 构建特征提取器：去掉classifier,保留feature + avgpool
	extractor = EfficientNetFeatureExtractor(model, pool=True, flatten=False)

	all_features = []
	all_labels = []

	print("Extracting Features...")
	with torch.no_grad():
		for images, labels in tqdm(train_loader, desc="Feature Extraction"):
			images = images.to(device)
			feats = extractor(images)   #[B, 1280, 1, 1]
			feats = feats.view(feats.size(0),-1).cpu().numpy()  #[B, 1280]
			all_features.append(feats)
			all_labels.append(labels.numpy())
			
	X = np.concatenate(all_features, axis=0)	# [N,1280]
	np.save("./Features/features_1280.npy",X)
	y = np.concatenate(all_labels, axis=0)	#[N,]

	print(f"Feature matrix shape:{X.shape}")

	# PCA降维
	from sklearn.decomposition import PCA
	print("Performing PCA to 128 demensions....")
	pca = PCA(n_components=128, svd_solver="full")
	X_reduced = pca.fit_transform(X)
	print(f"Reduced feature shape: {X_reduced.shape}")
	print(f"Explained variance ratio (top 5): {pca.explained_variance_ratio_[:5]}")
	print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
	# 保存结果
	save_dir = "./results"
	np.save(os.path.join(save_dir, "features_pca128.npy"), X_reduced)
	np.save(os.path.join(save_dir, "labels.npy"), y)

	# PCA模型
	import joblib
	joblib.dump(pca, os.path.join(save_dir, "pca_model.pth"))
	print("Features and PCA model saved.")

if __name__ == "__main__":

	# 如果你想只运行训练，注释掉下面这行
	# main()
	
	# 如果你想提取特征，取消注释下面这行
	extract_and_reduce_features()
