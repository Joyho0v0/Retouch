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


def split_list(items, seed):
    random.seed(seed)
    random.shuffle(items)
    total = len(items)
    train_count = int(total * 0.7)
    val_count = int(total * 0.2)
    test_count = total - train_count - val_count
    train_list = items[:train_count]
    val_list = items[train_count:train_count + val_count]
    test_list = items[train_count + val_count:]
    return train_list, val_list, test_list


def move_files(file_list, target_dir):
    ensure_dir(target_dir)
    for src_path in file_list:
        base_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, base_name)
        if os.path.exists(dst_path):
            name_part, ext = os.path.splitext(base_name)
            base_name = name_part + "_copy" + ext
            dst_path = os.path.join(target_dir, base_name)
        shutil.move(src_path, dst_path)


def rebuild_labels(split_dir, label_value):
    class_dir = os.path.join(split_dir, str(label_value))
    records = []
    for root, dirs, files in os.walk(class_dir):
        for name in files:
            lower_name = name.lower()
            if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".png") or lower_name.endswith(".bmp"):
                records.append((os.path.join(root, name), label_value))
    return records


def main():
    base_dir = "E:\\Project\\Re\\Retouch\\dataset"
    test_dir = os.path.join(base_dir, "test")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    test_zero = os.path.join(test_dir, "0")
    test_one = os.path.join(test_dir, "1")

    zero_images = collect_images(test_zero)
    one_images = collect_images(test_one)

    zero_train, zero_val, zero_test = split_list(zero_images, 123)
    one_train, one_val, one_test = split_list(one_images, 456)

    move_files(zero_train, os.path.join(train_dir, "0"))
    move_files(zero_val, os.path.join(val_dir, "0"))
    move_files(zero_test, os.path.join(test_dir, "0"))

    move_files(one_train, os.path.join(train_dir, "1"))
    move_files(one_val, os.path.join(val_dir, "1"))
    move_files(one_test, os.path.join(test_dir, "1"))

    train_records = rebuild_labels(train_dir, 0) + rebuild_labels(train_dir, 1)
    val_records = rebuild_labels(val_dir, 0) + rebuild_labels(val_dir, 1)
    test_records = rebuild_labels(test_dir, 0) + rebuild_labels(test_dir, 1)

    write_labels_file(os.path.join(train_dir, "labels.csv"), train_records)
    write_labels_file(os.path.join(val_dir, "labels.csv"), val_records)
    write_labels_file(os.path.join(test_dir, "labels.csv"), test_records)

    print("Resplit completed for test data.")
    print("labels.csv updated in train, val, test.")


if __name__ == "__main__":
    main()
