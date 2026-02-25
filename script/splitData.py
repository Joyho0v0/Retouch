import os
import random
import shutil


def collect_images(folder_path):
	image_list = []
	for root, dirs, files in os.walk(folder_path):
		for name in files:
			lower_name = name.lower()
			if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".png") or lower_name.endswith(".bmp"):
				image_list.append(os.path.join(root, name))
	return image_list


def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def write_labels_file(label_path, records):
	with open(label_path, "w", encoding="utf-8") as f:
		f.write("image_path,label\n")
		for item in records:
			f.write(item[0] + "," + str(item[1]) + "\n")


def split_and_copy(source_dir, target_root, label_value, seed):
	images = collect_images(source_dir)
	random.seed(seed)
	random.shuffle(images)

	total = len(images)
	train_count = int(total * 0.7)
	val_count = int(total * 0.2)
	test_count = total - train_count - val_count

	train_list = images[:train_count]
	val_list = images[train_count:train_count + val_count]
	test_list = images[train_count + val_count:]

	split_map = {
		"train": train_list,
		"val": val_list,
		"test": test_list
	}

	all_records = {
		"train": [],
		"val": [],
		"test": []
	}

	for split_name in split_map:
		target_dir = os.path.join(target_root, split_name, str(label_value))
		ensure_dir(target_dir)
		for src_path in split_map[split_name]:
			base_name = os.path.basename(src_path)
			dst_path = os.path.join(target_dir, base_name)
			if os.path.exists(dst_path):
				name_part, ext = os.path.splitext(base_name)
				base_name = name_part + "_copy" + ext
				dst_path = os.path.join(target_dir, base_name)
			shutil.copy2(src_path, dst_path)
			all_records[split_name].append((dst_path, label_value))

	return all_records


def main():
	base_dir = "./dataset"
	source_zero = os.path.join(base_dir, "megvii_total", "0")
	source_one = os.path.join(base_dir, "megvii_total", "1")

	target_root = "./dataset/megvii"
	train_dir = os.path.join(target_root, "train")
	val_dir = os.path.join(target_root, "val")
	test_dir = os.path.join(target_root, "test")

	ensure_dir(train_dir)
	ensure_dir(val_dir)
	ensure_dir(test_dir)

	seed = 42

	records_zero = split_and_copy(source_zero, target_root, 0, seed)
	records_one = split_and_copy(source_one, target_root, 1, seed + 1)

	for split_name in ["train", "val", "test"]:
		all_records = records_zero[split_name] + records_one[split_name]
		label_path = os.path.join(target_root, split_name, "labels.csv")
		write_labels_file(label_path, all_records)

	print("Split completed:")
	print("Train, Val, Test folders created with labels.csv")


if __name__ == "__main__":
	main()
