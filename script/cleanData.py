import random
import os

def move_half_images(src_folder, tmp_folder, exts=(".png", ".jpg", ".jpeg")):
	os.makedirs(tmp_folder, exist_ok=True)
	files = [f for f in os.listdir(src_folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(src_folder, f))]
	if not files:
		print("No images found.")
		return
	random.shuffle(files)
	half = len(files) // 2
	for f in files[:half]:
		src = os.path.join(src_folder, f)
		dst = os.path.join(tmp_folder, f)
		os.rename(src, dst)
	print(f"Moved {half} images to {tmp_folder}")

if __name__ == "__main__":
	# ...existing code...
	# 随机移动一半图片到tmp
	src_folder = r"E:\Project\Re\Retouch\dataset\test\0"
	tmp_folder = os.path.join(src_folder, "tmp")
	move_half_images(src_folder, tmp_folder)
# import shutil
# import os

# def move_conflict_images(conflict_txt, target_folder):
# 	os.makedirs(target_folder, exist_ok=True)
# 	with open(conflict_txt, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			line = line.strip()
# 			if not line:
# 				continue
# 			parts = line.split('\t')
# 			if len(parts) != 2:
# 				continue
# 			src, dst = parts
# 			img_path = dst.strip()
# 			if os.path.isfile(img_path):
# 				shutil.move(img_path, os.path.join(target_folder, os.path.basename(img_path)))

# if __name__ == "__main__":
# 	# ...existing code...
# 	# 移动conflict图片
# 	conflict_txt = r"E:\Project\Re\Retouch\dataset\test\problem\conflict.txt"
# 	target_folder = r"E:\Project\Re\Retouch\dataset\test\problem"
# 	move_conflict_images(conflict_txt, target_folder)
# import os

# def find_intersection(folder1, folder2, exts=(".png", ".jpg", ".jpeg")):
# 	files1 = set(f for f in os.listdir(folder1) if f.lower().endswith(exts))
# 	files2 = set(f for f in os.listdir(folder2) if f.lower().endswith(exts))
# 	return files1 & files2

# if __name__ == "__main__":
# 	total_dir = r"E:\Project\Re\Retouch\dataset\total"
# 	test0_dir = r"E:\Project\Re\Retouch\dataset\test\0"
# 	intersection = find_intersection(total_dir, test0_dir)
# 	print(f"交集图片数量: {len(intersection)}")
# 	for name in sorted(intersection):
# 		print(name)
# # -*- coding: utf-8 -*-
# import os
# import shutil
# import ast

# TXT_PATHS = [
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_ali_process\one_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_dual_process\dual_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_dual_process\dual_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_dual_process\two_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_process\one_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\three_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_three_process\three_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_four_process\four_process.txt",
# 	r"E:\Project\Re\Retouch\dataset\FFHQ_four_process\four_process.txt",
# ]

# def read_problem_names(txt_paths):
# 	problem_names = set()
# 	for txt_path in txt_paths:
# 		if not os.path.exists(txt_path):
# 			continue
# 		with open(txt_path, 'r', encoding='utf-8') as f:
# 			for line in f:
# 				line = line.strip()
# 				if not line:
# 					continue
# 				parts = line.split('\t')
# 				if len(parts) != 2:
# 					continue
# 				dict_str, img_path = parts
# 				try:
# 					op_dict = ast.literal_eval(dict_str)
# 				except Exception:
# 					continue
# 				if isinstance(op_dict, dict) and op_dict.get('EyeEnlarging', None) == 0:
# 					img_name = os.path.basename(img_path)
# 					problem_names.add(img_name)
# 	return problem_names

# def list_all_images(folder):
# 	imgs = []
# 	for root, dirs, files in os.walk(folder):
# 		for f in files:
# 			if f.lower().endswith(('.png', '.jpg', '.jpeg')):
# 				imgs.append(os.path.join(root, f))
# 	return imgs

# def move_images(src_folder, dst_folder, problem_names, total_folder, conflict_txt):
# 	os.makedirs(dst_folder, exist_ok=True)
# 	os.makedirs(total_folder, exist_ok=True)
# 	conflict_path = os.path.join(total_folder, conflict_txt)
# 	written = set(os.listdir(total_folder))
# 	with open(conflict_path, 'a', encoding='utf-8') as cf:
# 		imgs = list_all_images(src_folder)
# 		for img_path in imgs:
# 			img_name = os.path.basename(img_path)
# 			if img_name in problem_names:
# 				dst_path = os.path.join(dst_folder, img_name)
# 				shutil.move(img_path, dst_path)
# 			else:
# 				dst_path = os.path.join(total_folder, img_name)
# 				if img_name in written:
# 					cf.write(f"{img_path}\t{dst_path}\n")
# 					continue
# 				shutil.move(img_path, dst_path)
# 				written.add(img_name)

# def main():
# 	txt_paths = TXT_PATHS
# 	src_folder = r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\Smoothing_FaceLifting_EyeEnlarging"
# 	test0_folder = r"E:\Project\Re\Retouch\dataset\test\0"
# 	total_folder = r"E:\Project\Re\Retouch\dataset\total"
# 	conflict_txt = "conflict.txt"

# 	# 1. 读取所有txt筛选问题图片名
# 	problem_names = read_problem_names(txt_paths)

# 	# 2. 处理Smoothing_FaceLifting_EyeEnlarging下所有图片
# 	# 问题图片移到test/0，正常图片移到total，冲突记录
# 	move_images(src_folder, test0_folder, problem_names, total_folder, conflict_txt)

# if __name__ == '__main__':
# 	main()
# # -*- coding: utf-8 -*-
# import os
# import shutil
# import ast

# def read_fail_txt(fail_txt_path):
# 	result = set()
# 	with open(fail_txt_path, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			line = line.strip()
# 			if not line:
# 				continue
# 			parts = line.split('\t')
# 			if len(parts) != 2:
# 				continue
# 			op, img_path = parts
# 			if 'EyeEnlarging' in op:
# 				path_parts = img_path.replace('\\', '/').split('/')
# 				if len(path_parts) >= 2 and path_parts[-2] == '20000':
# 					result.add(os.path.basename(img_path))
# 	return result

# def read_problem_names(three_process_path):
# 	problem_names = set()
# 	with open(three_process_path, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			line = line.strip()
# 			if not line:
# 				continue
# 			parts = line.split('\t')
# 			if len(parts) != 2:
# 				continue
# 			dict_str, img_path = parts
# 			try:
# 				op_dict = ast.literal_eval(dict_str)
# 			except Exception:
# 				continue
# 			if isinstance(op_dict, dict) and op_dict.get('EyeEnlarging', None) == 0:
# 				img_name = os.path.basename(img_path)
# 				problem_names.add(img_name)
# 	return problem_names

# def list_all_images(folder):
# 	imgs = []
# 	for root, dirs, files in os.walk(folder):
# 		for f in files:
# 			if f.lower().endswith(('.png', '.jpg', '.jpeg')):
# 				imgs.append(os.path.join(root, f))
# 	return imgs

# def move_images(src_folder, dst_folder, problem_names, total_folder, conflict_txt):
# 	os.makedirs(dst_folder, exist_ok=True)
# 	os.makedirs(total_folder, exist_ok=True)
# 	conflict_path = os.path.join(total_folder, conflict_txt)
# 	written = set(os.listdir(total_folder))
# 	with open(conflict_path, 'a', encoding='utf-8') as cf:
# 		imgs = list_all_images(src_folder)
# 		for img_path in imgs:
# 			img_name = os.path.basename(img_path)
# 			if img_name in problem_names:
# 				dst_path = os.path.join(dst_folder, img_name)
# 				shutil.move(img_path, dst_path)
# 			else:
# 				dst_path = os.path.join(total_folder, img_name)
# 				if img_name in written:
# 					cf.write(f"{img_path}\t{dst_path}\n")
# 					continue
# 				shutil.move(img_path, dst_path)
# 				written.add(img_name)

# def filter_total_by_excluded(total_folder, excluded_txt):
# 	excluded_names = set()
# 	with open(excluded_txt, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			name = line.strip()
# 			if name:
# 				excluded_names.add(name)
# 	for f in os.listdir(total_folder):
# 		if f.lower().endswith(('.png', '.jpg', '.jpeg')):
# 			if os.path.splitext(f)[0] in excluded_names:
# 				os.remove(os.path.join(total_folder, f))

# def main():
# 	fail_txt = r"E:\Project\Re\Retouch\dataset\FFHQ_dual_process\fail.txt"
# 	three_process_txt = r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\FFHQ_three_process\three_process.txt"
# 	src_folder = r"E:\Project\Re\Retouch\dataset\FFHQ_megvii_three_process\Smoothing_FaceLifting_EyeEnlarging"
# 	test0_folder = r"E:\Project\Re\Retouch\dataset\test\0"
# 	total_folder = r"E:\Project\Re\Retouch\dataset\total"
# 	conflict_txt = "conflict.txt"
# 	excluded_txt = r"E:\Project\Re\Retouch\dataset\FFHQ_settings\excluded_images_list\total.txt"

# 	# 1. 读取fail.txt筛选
# 	fail_basenames = read_fail_txt(fail_txt)

# 	# 2. 读取three_process.txt问题图片名
# 	problem_names = read_problem_names(three_process_txt)

# 	# 3. 处理Smoothing_FaceLifting_EyeEnlarging下所有图片
# 	# 问题图片移到test/0，正常图片移到total，冲突记录
# 	move_images(src_folder, test0_folder, problem_names, total_folder, conflict_txt)

# 	# 4. total文件夹内图片按excluded_images_list/total.txt排除
# 	filter_total_by_excluded(total_folder, excluded_txt)

# if __name__ == '__main__':
# 	main()
# # -*- coding: utf-8 -*-
# import os
# import shutil

# def read_fail_txt(fail_txt_path):
# 	result = set()
# 	with open(fail_txt_path, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			line = line.strip()
# 			if not line:
# 				continue
# 			parts = line.split('\t')
# 			if len(parts) != 2:
# 				continue
# 			op, img_path = parts
# 			if op == 'EyeEnlarging_30':
# 				# 倒数第二级目录
# 				path_parts = img_path.replace('\\', '/').split('/')
# 				if len(path_parts) >= 2 and path_parts[-2] == '17000':
# 					result.add(os.path.basename(img_path))
# 	return result

# def read_excluded_list(excluded_txt_path):
# 	excluded = set()
# 	with open(excluded_txt_path, 'r', encoding='utf-8') as f:
# 		for line in f:
# 			name = line.strip()
# 			if name:
# 				excluded.add(name)
# 	return excluded

# def list_images(folder):
# 	return set([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# def filter_images(folder, exclude_names, exclude_basenames):
# 	imgs = list_images(folder)
# 	filtered = [f for f in imgs if os.path.splitext(f)[0] not in exclude_names and f not in exclude_basenames]
# 	return filtered

# def merge_folders(src_folders, dst_folder, conflict_txt):
# 	os.makedirs(dst_folder, exist_ok=True)
# 	conflict_path = os.path.join(dst_folder, conflict_txt)
# 	written = set(os.listdir(dst_folder))
# 	with open(conflict_path, 'a', encoding='utf-8') as cf:
# 		for src in src_folders:
# 			for f in os.listdir(src):
# 				src_file = os.path.join(src, f)
# 				if not os.path.isfile(src_file):
# 					continue
# 				dst_file = os.path.join(dst_folder, f)
# 				if f in written:
# 					cf.write(f"{src_file}\t{dst_file}\n")
# 					continue
# 				shutil.copy2(src_file, dst_file)
# 				written.add(f)

# def main():
# 	fail_txt = r"E:\Project\Re\Retouch\dataset\FFHQ_ali_process\fail.txt"
# 	ali_17000 = r"E:\Project\Re\Retouch\dataset\FFHQ_ali_process\EyeEnlarging_30\17000"
# 	process_00000 = r"E:\Project\Re\Retouch\dataset\FFHQ_process\EyeEnlarging_30\00000"
# 	excluded_txt = r"E:\Project\Re\Retouch\dataset\FFHQ_settings\excluded_images_list\total.txt"
# 	total_folder = r"E:\Project\Re\Retouch\dataset\total"
# 	conflict_txt = "conflict.txt"

# 	# 1. 读取fail.txt筛选
# 	fail_basenames = read_fail_txt(fail_txt)

# 	# 2. 处理ali_17000，排除fail_basenames
# 	ali_imgs = filter_images(ali_17000, set(), fail_basenames)

# 	# 3. 读取问题图片名
# 	excluded_names = read_excluded_list(excluded_txt)

# 	# 4. 处理ali_17000和process_00000，排除问题图片
# 	ali_imgs = [f for f in ali_imgs if os.path.splitext(f)[0] not in excluded_names]
# 	process_imgs = filter_images(process_00000, excluded_names, set())

# 	# 5. 合并到total，记录冲突
# 	# 先将ali_imgs和process_imgs写入临时文件夹，再合并
# 	import tempfile
# 	with tempfile.TemporaryDirectory() as tmp:
# 		ali_tmp = os.path.join(tmp, 'ali')
# 		process_tmp = os.path.join(tmp, 'process')
# 		os.makedirs(ali_tmp)
# 		os.makedirs(process_tmp)
# 		for f in ali_imgs:
# 			shutil.copy2(os.path.join(ali_17000, f), os.path.join(ali_tmp, f))
# 		for f in process_imgs:
# 			shutil.copy2(os.path.join(process_00000, f), os.path.join(process_tmp, f))
# 		merge_folders([ali_tmp, process_tmp], total_folder, conflict_txt)

# if __name__ == '__main__':
# 	main()
