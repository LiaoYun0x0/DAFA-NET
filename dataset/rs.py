import os
import io
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

# --- 移植自 uw.py: 确定的 JPEG 压缩变换 ---
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


# --- 修改后的 RSFakeDataset (参考 UWDataset 格式) ---
class RSFakeDataset(Dataset):
    """
    用于 RSFAKE-1M 数据集的数据加载类。
    """

    def __init__(self, root_dir, split='train', blur_sigma=None, jpeg_quality=None, augment_other=False, sample_ratio=1.0):
        """
        Args:
            root_dir (string): RSFAKE-1M 数据集的根目录路径。
            split (string): 'train', 'val', 或 'test'。
            blur_sigma (float, optional): 如果提供，将对图像应用高斯模糊。
            jpeg_quality (int, optional): 如果提供，将对图像应用 JPEG 压缩。
            augment_other (bool): 是否应用其他标准增强 (随机翻转)。
            sample_ratio (float): 数据采样比例 (0.0 < ratio <= 1.0)，默认 1.0 (使用全部数据)。
        """
        self.root_dir = root_dir
        self.split = split

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

        # 7. 根据 split 选择对应的 CSV 文件
        csv_file = os.path.join(root_dir, 'SPLIT', f'RSFAKE_{split}_new.csv')

        # 8. 读取 CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到 CSV 文件: {csv_file}")
        self.data_info = pd.read_csv(csv_file)

        # 【新增】核心修改：数据采样逻辑
        if sample_ratio < 1.0:
            total_before = len(self.data_info)
            # frac=sample_ratio 表示按比例随机抽样
            # random_state=42 保证每次取的数据都是同一批 20%
            self.data_info = self.data_info.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
            print(
                f"[{split}] 已启用采样: 从 {total_before} 张减少到 {len(self.data_info)} 张 (保留 {sample_ratio * 100}%)")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 获取图片相对路径和标签
        # 假设 CSV 列名是 'image_path' 和 'label'
        try:
            img_rel_path = self.data_info.iloc[idx]['image_path']
            label = int(self.data_info.iloc[idx]['label'])
        except KeyError:
            # 如果没有表头，尝试用索引读取
            img_rel_path = self.data_info.iloc[idx, 0]
            label = int(self.data_info.iloc[idx, 1])

        # 拼接绝对路径
        img_path = os.path.join(self.root_dir, img_rel_path)

        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error: 无法加载图像 {img_path}. {e}")
            # 如果图片坏了，为了不报错中断，可以返回一张全黑图或者跳过（简单起见这里返回黑图）
            image = Image.new('RGB', (256, 256))
            # return None, None  # 返回空信号

        # 应用在 __init__ 中定义的 transform 流程
        image = self.transform(image)

        return image, label

    # 定义静态过滤函数 (处理损坏图片)
    @staticmethod
    def collate_fn_skip_none(batch):
        # 过滤掉 None 的项
        batch = [item for item in batch if item[0] is not None]

        if len(batch) == 0:
            return None

        return default_collate(batch)


# --- 标准化测试模块 (参考 uw.py main 风格) ---
if __name__ == '__main__':

    # 设置路径 (请根据实际情况修改)
    root = r"E:\data\RSFAKE-1M\RSFAKE-1M"

    # 1. 创建数据集实例 (设置确定的变换)
    print(f"--- 正在加载 RSFAKE 数据集 (split='train') ---")

    # 例如：应用模糊和压缩进行鲁棒性训练/测试
    dataset_with_transforms = RSFakeDataset(
        root_dir=root,
        split='train',
        blur_sigma=1.5,  # 示例：设置模糊强度
        jpeg_quality=80  # 示例：设置压缩强度
    )

    total_images = len(dataset_with_transforms)
    print(f"\n成功加载数据集。总图像数: {total_images}")

    if total_images > 0:

        # 2. 创建 DataLoader
        # 注意：这里传入了 collate_fn 以处理可能的坏图
        data_loader = DataLoader(
            dataset=dataset_with_transforms,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            collate_fn=RSFakeDataset.collate_fn_skip_none
        )

        # 3. 从 DataLoader 中获取一个批次的数据进行测试
        print(f"\n--- 测试 DataLoader (Batch Size = 8) ---")
        try:
            batch = next(iter(data_loader))

            if batch is None:
                print("⚠️ Batch 为空 (所有图片均损坏)")
            else:
                images, labels = batch
                print(f"成功读取一个 Batch:")
                print(f" - 图片 Tensor 形状: {images.shape}")
                print(f" - 标签 Tensor: {labels}")
                print(f" - 标签示例: {labels.tolist()}")

        except Exception as e:
            print(f"❌ 读取 Batch 失败: {e}")
            import traceback

            traceback.print_exc()
