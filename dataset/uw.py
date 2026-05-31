import os
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# --- 新增：确定的 JPEG 压缩变换 ---
class ApplyJPEGCompression:
    """
    确定性地（100%）对图像应用指定质量的JPEG压缩。
    """

    def __init__(self, quality):
        """
        Args:
            quality (int): 压缩质量 (1-100)。
        """
        self.quality = int(quality)
        if not (1 <= self.quality <= 100):
            raise ValueError("JPEG quality must be between 1 and 100.")

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 需要变换的图像。

        Returns:
            PIL Image: 变换后的图像。
        """
        # 创建一个内存中的字节缓冲区
        buffer = io.BytesIO()

        # 以指定的JPEG质量将图像保存到缓冲区
        img.save(buffer, "JPEG", quality=self.quality)

        # 从缓冲区重新打开图像
        img = Image.open(buffer)

        return img


# --- 更新后的 UWDataset ---
class UWDataset(Dataset):
    """
    用于 "Deep Fake Geography" UW 数据集的数据加载类。
    """

    def __init__(self, root_dir, blur_sigma=None, jpeg_quality=None, augment_other=False):
        """
        Args:
            root_dir (string): 'data' 文件夹的路径。
            blur_sigma (float, optional): 高斯模糊的 sigma 值。
                如果设置 (e.g., 1.5)，将 100% 应用此模糊。
            jpeg_quality (int, optional): JPEG 压缩质量 (1-100)。
                如果设置 (e.g., 80)，将 100% 应用此压缩。
            augment_other (bool): 是否应用其他标准增强 (随机翻转)。
        """
        self.root_dir = root_dir
        self.samples = []  # 存储 (image_path, label) 元组的列表
        self.classes = {'authentic': 0, 'fake': 1}

        # --- 内部构建 Transform 流程 ---

        # 1. 基础变换 (所有图像都需要)
        transform_pipeline = [
            transforms.Resize((256, 256))
        ]

        # 2. 添加其他（随机）增强
        if augment_other:
            print("Applying random augmentations (flips).")
            transform_pipeline.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])

        # 3. 添加确定的高斯模糊 (如果 blur_sigma 被设置)
        if blur_sigma is not None and blur_sigma > 0:
            # 使用一个固定的 kernel_size，例如 7x7。
            # sigma 才是控制模糊“强度”的关键。
            kernel_size = 7
            print(f"Applying DETERMINISTIC Gaussian Blur (sigma={blur_sigma})")
            transform_pipeline.append(
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=blur_sigma)
            )

        # 4. 添加确定的JPEG压缩 (如果 jpeg_quality 被设置)
        if jpeg_quality is not None:
            print(f"Applying DETERMINISTIC JPEG Compression (quality={jpeg_quality})")
            transform_pipeline.append(
                ApplyJPEGCompression(quality=jpeg_quality)
            )

        # 5. 最终变换 (转为Tensor和归一化)
        transform_pipeline.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 6. 创建最终的 transform
        self.transform = transforms.Compose(transform_pipeline)

        # --- 结束 Transform 流程 ---

        # 调用方法加载数据路径
        self._load_dataset()

    def _load_dataset(self):
        """遍历文件夹结构，收集所有图像路径和标签。"""

        for class_name, label in self.classes.items():
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_dir):
                print(f"Warning: 目录 {class_dir} 不存在。")
                continue

            for sub_folder_name in os.listdir(class_dir):
                sub_folder_path = os.path.join(class_dir, sub_folder_name)

                if not os.path.isdir(sub_folder_path):
                    continue

                for img_name in os.listdir(sub_folder_path):
                    if img_name.lower().endswith('.png') or img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(sub_folder_path, img_name)
                        self.samples.append((img_path, label))

        if not self.samples:
            print(f"Error: 在 {self.root_dir} 中没有找到任何图像。请检查路径。")

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签。
        """
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error: 无法加载图像 {img_path}. {e}")
            return None, None

        # 应用在 __init__ 中定义的 transform 流程
        image = self.transform(image)

        return image, label


# --- 以下是示例用法 ---
if __name__ == '__main__':

    data_directory = 'data'

    # 1. 创建数据集实例 (设置确定的变换)
    print(f"--- 正在加载数据集 (带特定变换) ---")

    # 例如：
    # - 模糊强度 sigma=1.5
    # - 压缩强度 quality=80
    dataset_with_transforms = UWDataset(
        root_dir=data_directory,
        blur_sigma=1.5,  # <--- 在这里设置模糊强度
        jpeg_quality=80,  # <--- 在这里设置压缩强度
        augment_other=False  # (关闭随机翻转，以保持一致性)
    )

    total_images = len(dataset_with_transforms)
    print(f"\n成功加载数据集。总图像数: {total_images}")

    if total_images > 0:

        # 2. 创建 DataLoader
        data_loader = DataLoader(
            dataset=dataset_with_transforms,
            batch_size=8,
            shuffle=False,  # 关闭 shuffle 以便观察
            num_workers=0
        )

        # 3. 从 DataLoader 中获取一个批次的数据
        print(f"\n--- 测试 DataLoader (Batch Size = 8) ---")
        try:
            images, labels = next(iter(data_loader))

            print(f"成功获取一个批次。")
            print(f"图像批次 (Images) 的形状: {images.shape}")
            print(f"标签批次 (Labels) 的形状: {labels.shape}")

        except Exception as e:
            print(f"加载 DataLoader 失败: {e}")

    # --- 示例 2：不带任何模糊和压缩 ---
    print(f"\n--- 正在加载数据集 (不带变换) ---")

    dataset_no_transforms = UWDataset(
        root_dir=data_directory,
        blur_sigma=None,
        jpeg_quality=None
    )
    print(f"\n成功加载数据集。总图像数: {len(dataset_no_transforms)}")
