import os
import pickle
import pprint

import numpy as np
import matplotlib.pyplot as plt


def summarize_obj(obj, max_depth=3, max_items=10, indent=0):
	"""简单打印 pkl 结构信息，避免一次性打印太多。"""
	prefix = " " * indent
	if indent // 2 >= max_depth:
		print(prefix + f"<max_depth reached, type={type(obj)}>\n")
		return

	if isinstance(obj, dict):
		print(prefix + f"dict (len={len(obj)}):")
		for i, (k, v) in enumerate(obj.items()):
			if i >= max_items:
				print(prefix + f"  ... ({len(obj) - max_items} more keys)")
				break
			print(prefix + f"  key={k!r}, type={type(v)}")
			if isinstance(v, (list, tuple, np.ndarray)):
				try:
					arr = np.array(v)
					print(prefix + f"    shape={arr.shape}, dtype={arr.dtype}")
				except Exception:
					pass
			elif isinstance(v, dict):
				# 只看一层
				print(prefix + f"    inner-dict len={len(v)}")
	elif isinstance(obj, (list, tuple)):
		print(prefix + f"{type(obj).__name__} (len={len(obj)}):")
		for i, v in enumerate(obj[:max_items]):
			print(prefix + f"  idx={i}, type={type(v)}")
		print(prefix + "  ...")
	elif isinstance(obj, np.ndarray):
		print(prefix + f"ndarray shape={obj.shape}, dtype={obj.dtype}")
	else:
		print(prefix + f"{type(obj)}: {repr(obj)[:200]}")


def try_plot_selector(selector, out_dir, base_name):
	"""如果是通道选择器风格的 dict，尝试画 NMI 曲线。"""
	if not isinstance(selector, dict):
		return

	os.makedirs(out_dir, exist_ok=True)

	# 1) 直接 nmi_curve
	if "nmi_curve" in selector:
		try:
			y = np.array(selector["nmi_curve"], dtype=float)
			x = np.arange(1, len(y) + 1)
			plt.figure()
			plt.plot(x, y, marker="o")
			plt.xlabel("k (num channels)")
			plt.ylabel("NMI")
			plt.title("NMI curve from selector.nmi_curve")
			plt.grid(True, linestyle="--", alpha=0.5)
			out_path = os.path.join(out_dir, f"{base_name}_nmi_curve.png")
			plt.savefig(out_path, dpi=150, bbox_inches="tight")
			plt.close()
			print(f"保存 NMI 曲线: {out_path}")
		except Exception as e:
			print(f"画 nmi_curve 时出错: {e}")

	# 2) checkpoint_results 里可能也有 NMI 信息
	chk = selector.get("checkpoint_results")
	if isinstance(chk, dict) and len(chk) > 0:
		# 期望形如 {k: {"nmi": ..., ...}, ...}
		ks = []
		nmis = []
		for k, v in chk.items():
			try:
				kk = int(k)
			except Exception:
				continue
			if isinstance(v, dict):
				val = None
				# 常见几种 key 名称尝试一下
				for key in ["nmi", "kmeans_nmi", "subspace_nmi"]:
					if key in v:
						val = v[key]
						break
				if val is None:
					continue
				try:
					val = float(val)
				except Exception:
					continue
				ks.append(kk)
				nmis.append(val)

		if ks:
			idx = np.argsort(ks)
			ks = np.array(ks)[idx]
			nmis = np.array(nmis)[idx]
			plt.figure()
			plt.plot(ks, nmis, marker="o")
			plt.xlabel("k (num channels)")
			plt.ylabel("NMI (from checkpoint_results)")
			plt.title("NMI curve from selector.checkpoint_results")
			plt.grid(True, linestyle="--", alpha=0.5)
			out_path = os.path.join(out_dir, f"{base_name}_checkpoint_nmi.png")
			plt.savefig(out_path, dpi=150, bbox_inches="tight")
			plt.close()
			print(f"保存 checkpoint NMI 曲线: {out_path}")


def main():
	# 直接在这里改路径即可
	pkl_path = "./nmi_channel_selector.pkl"  # 你要查看的 pkl 文件
	out_dir = "./results/view_pkl"           # 可视化图片保存目录

	if not os.path.exists(pkl_path):
		print(f"文件不存在: {pkl_path}")
		return

	# 用于输出文件命名，例如 nmi_channel_selector_dict.txt
	base_name = os.path.splitext(os.path.basename(pkl_path))[0]

	print("加载 pkl: " + pkl_path)
	with open(pkl_path, "rb") as f:
		obj = pickle.load(f)

	print("\n===== 基本信息 =====")
	print(f"type: {type(obj)}")
	try:
		if isinstance(obj, np.ndarray):
			print(f"shape: {obj.shape}, dtype: {obj.dtype}")
		elif isinstance(obj, (list, tuple)):
			print(f"len: {len(obj)}")
		elif isinstance(obj, dict):
			print(f"len(dict.keys): {len(obj)}")
	except Exception:
		pass

	print("\n===== 结构摘要 =====")
	summarize_obj(obj)

	# 如果是 dict，把完整内容写入 txt
	if isinstance(obj, dict):
		os.makedirs(out_dir, exist_ok=True)
		dict_txt_path = os.path.join(out_dir, f"{base_name}_dict.txt")
		with open(dict_txt_path, "w", encoding="utf-8") as f:
			f.write(f"pkl_path: {pkl_path}\n")
			f.write(f"type: {type(obj)}\n")
			f.write(f"len(dict.keys): {len(obj)}\n")
			f.write("\n===== FULL DICT CONTENT (key -> value) =====\n\n")
			for i, (k, v) in enumerate(obj.items()):
				f.write(f"[{i}] KEY: {repr(k)}\n")
				f.write("VALUE:\n")
				try:
					text = pprint.pformat(v, width=120, compact=False)
				except Exception:
					text = repr(v)
				f.write(text)
				f.write("\n" + "-" * 80 + "\n\n")
		print(f"已将完整字典内容写入: {dict_txt_path}")

	# 如果是通道选择器风格的 dict，尝试画 NMI 曲线
	try_plot_selector(obj, out_dir, base_name)

	print("\n查看完成。若生成了图片，可到 out-dir 里查看。")


if __name__ == "__main__":
	main()
