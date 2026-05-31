import os
import shutil
import random
import math


def copy_files_to_splits(source_path, dest_paths, ratios, split_names):
    """
    辅助函数：从源路径复制文件到 train/val/test 目标路径。

    Args:
        source_path (str): 图像所在的源文件夹。
        dest_paths (dict): 包含 'train', 'val', 'test' 键的目标路径字典。
        ratios (tuple): (train, val, test) 比例。
        split_names (list): ['train', 'val', 'test']
    """
    try:
        # 允许 .png, .jpg, .jpeg
        images = [
            f for f in os.listdir(source_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        random.shuffle(images)  # 随机打乱
    except Exception as e:
        print(f"    Error reading {source_path}: {e}")
        return

    total_count = len(images)
    if total_count == 0:
        print(f"    No images found in {source_path}")
        return

    # 5. 计算分割点
    train_count = int(total_count * ratios[0])
    val_count = int(total_count * ratios[1])
    # test_count = total_count - train_count - val_count

    # 7. 复制文件
    # 训练集
    file_list = images[:train_count]
    for img in file_list:
        shutil.copy(os.path.join(source_path, img), dest_paths[split_names[0]])

    # 验证集
    file_list = images[train_count: train_count + val_count]
    for img in file_list:
        shutil.copy(os.path.join(source_path, img), dest_paths[split_names[1]])

    # 测试集
    file_list = images[train_count + val_count:]
    for img in file_list:
        shutil.copy(os.path.join(source_path, img), dest_paths[split_names[2]])

    print(
        f"    Total: {total_count} | Train: {train_count}, Val: {val_count}, Test: {total_count - train_count - val_count}")


def split_files(source_dir, dest_dir, ratios=(0.7, 0.2, 0.1), seed=42):
    """
    将源目录中的图像按比例划分到目标目录的 train, val, test 文件夹中。

    Args:
        source_dir (str): 'data' 文件夹路径.
        dest_dir (str): 'data_split' 文件夹路径 (将自动创建).
        ratios (tuple): (train, val, test) 比例.
        seed (int): 随机种子，确保可复现.
    """

    # 修正浮点数精度问题
    assert abs(sum(ratios) - 1.0) < 0.000001, "Ratios must sum to 1.0"

    random.seed(seed)

    # 定义 train, val, test 文件夹名称
    split_names = ['train', 'val', 'test']

    # 1. 创建目标目录结构
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    for split_name in split_names:
        split_path = os.path.join(dest_dir, split_name)
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    # 2. 遍历源目录 (authentic, fake)
    for top_folder in os.listdir(source_dir):
        source_top_path = os.path.join(source_dir, top_folder)

        if not os.path.isdir(source_top_path):
            continue

        print(f"\nProcessing folder: {top_folder}")

        # 在 train/val/test 下创建 authentic/fake 文件夹
        # e.g., data_split/train/fake/
        dest_top_paths = {}
        for split_name in split_names:
            path = os.path.join(dest_dir, split_name, top_folder)
            if not os.path.exists(path):
                os.makedirs(path)
            dest_top_paths[split_name] = path

        # --- 核心逻辑：检查是“嵌套”还是“扁平” ---
        items_in_top_folder = os.listdir(source_top_path)

        # 检查是否包含子目录
        has_subdirs = False
        for item in items_in_top_folder:
            if os.path.isdir(os.path.join(source_top_path, item)):
                has_subdirs = True
                break

        # --- Case 1: 嵌套结构 (如 'authentic') ---
        if has_subdirs:
            print(f"  '{top_folder}' contains sub-folders. Processing nested structure.")
            for sub_folder in items_in_top_folder:
                source_sub_path = os.path.join(source_top_path, sub_folder)

                if not os.path.isdir(source_sub_path):
                    continue  # 跳过 .DS_Store 之类的文件

                print(f"    Processing sub-folder: {sub_folder}")

                # 创建目标子目录
                # e.g., data_split/train/authentic/Beijing/
                dest_sub_paths = {}
                for split_name in split_names:
                    path = os.path.join(dest_top_paths[split_name], sub_folder)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    dest_sub_paths[split_name] = path

                # 复制文件
                copy_files_to_splits(source_sub_path, dest_sub_paths, ratios, split_names)

        # --- Case 2: 扁平结构 (可能如 'fake') ---
        else:
            print(f"  '{top_folder}' is flat (no sub-folders). Processing files directly.")
            # 目标路径就是顶层路径
            # e.g., data_split/train/fake/
            copy_files_to_splits(source_top_path, dest_top_paths, ratios, split_names)

    print("\n--- Split complete! ---")
    print(f"Data has been copied to: {dest_dir}")


# --- 运行脚本 ---
if __name__ == '__main__':
    SOURCE_DATA_DIR = 'D://z_twb//Data//anti-deepfake-data and code//data'
    DEST_DATA_DIR = 'D://z_twb//Data//anti-deepfake-data and code//data_split'

    # 7:2:1 比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    if os.path.exists(DEST_DATA_DIR):
        print(f"Warning: Destination folder '{DEST_DATA_DIR}' already exists.")
        print("Please delete it first if you want a clean split.")
        # input("Press Enter to continue (or Ctrl+C to cancel)...")

    split_files(SOURCE_DATA_DIR, DEST_DATA_DIR, ratios=(TRAIN_RATIO, VAL_RATIO, TEST_RATIO))