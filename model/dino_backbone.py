import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# --- Python 路径处理 ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 dinov3
try:
    from dinov3.hub.backbones import (
        dinov3_vits16, dinov3_vitb16, dinov3_vitl16,
        dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base
    )
except ImportError:
    try:
        from model.dinov3.hub.backbones import (
            dinov3_vits16, dinov3_vitb16, dinov3_vitl16,
            dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base
        )
    except ImportError as e:
        print(f"错误: 无法导入 dinov3 包。\n详情: {e}")
        sys.exit(1)


# --- 封装的dinov3模型类 ---
class MyDinoV3Model(nn.Module):
    """
        需要的输入 (Input): (Batch_Size, 3, Height, Width)
            Channels: 3 (RGB 图像)
            Height, Width: 最好是 14 (patch_size) 或 16 (默认 patch_size) 的倍数。
            虽然模型可以处理任意尺寸（只要大于 patch size），但标准预训练尺寸通常是 224x224 或 518x518。
            注意: 代码中已经加入了 Resize 到 224 的逻辑
        输出的特征尺寸:
            经过 output = self.backbone(x) 后：数据类型: torch.Tensor (注意：不是字典)
            特征维度: (Batch_Size, 768), 768 是 ViT-Base 的嵌入维度 (embed_dim)。
            Batch_Size 与输入的图片数量一致
    """
    def __init__(
            self,
            weights_path: str,
            num_classes: int = 10,
            model_name: str = 'base',
            freeze_backbone: bool = True,
            img_size: int = 224
    ):
        super().__init__()
        self.img_size = img_size

        # 1. 模型选择器
        model_dict = {
            'small': dinov3_vits16,
            'base': dinov3_vitb16,
            'large': dinov3_vitl16,
            'convnext_tiny': dinov3_convnext_tiny,
            'convnext_small': dinov3_convnext_small,
            'convnext_base': dinov3_convnext_base,
        }

        if model_name not in model_dict:
            raise ValueError(f"不支持的模型名称: {model_name}")

        backbone_fn = model_dict[model_name]

        # 2. 加载主干
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到权重文件: {weights_path}")

        print(f"正在加载 DINOv3 ({model_name}) ...")
        self.backbone = backbone_fn(weights=weights_path)

        # 3. 冻结主干
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 4. 自动获取特征维度
        with torch.no_grad():
            # 创建一个假输入来探测输出维度
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)

            # 兼容性处理：判断返回是字典还是Tensor
            if isinstance(out, dict):
                feat_dim = out['x_norm_clstoken'].shape[-1]
            elif isinstance(out, torch.Tensor):
                feat_dim = out.shape[-1]
            else:
                raise TypeError(f"未知的模型输出类型: {type(out)}")

        print(f"检测到特征维度: {feat_dim}")

        # 5. 分类头
        self.classifier_head = nn.Linear(feat_dim, num_classes)

        # 初始化分类头
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

    def get_intermediate_layers(self, x, n=[2, 4, 7, 11], reshape=True, norm=True):
        # 自动调整大小 (针对 CIFAR-10 等小图)
        if x.shape[-1] < self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)

        # 提取指定层的特征图
        # reshape=True: 返回 (B, C, H, W) 格式
        # norm=True: 对提取的特征进行 LayerNorm (推荐)
        # feature_maps 是一个包含 4 个 Tensor 的 tuple
        # 每个 Tensor 的形状是 (Batch, 768, H/16, W/16)
        feature_maps = self.backbone.get_intermediate_layers(
            x,
            n=n,  # 对应第 3, 5, 8, 12 层
            reshape=reshape,
            norm=norm
        )
        return feature_maps

    def forward(self, x):
        # 自动调整大小 (针对 CIFAR-10 等小图)
        if x.shape[-1] < self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)

        # 提取特征
        if not self.backbone.training:
            with torch.no_grad():
                output = self.backbone(x)
        else:
            output = self.backbone(x)

        # --- !!! 修复点在此 !!! ---
        # 判断 output 是字典还是 Tensor
        if isinstance(output, dict):
            features = output["x_norm_clstoken"]
        else:
            # 如果不是字典，DINOv3 默认直接返回了 CLS Token
            features = output

        # 分类
        logits = self.classifier_head(features)
        return logits


# --- 测试代码 ---
if __name__ == "__main__":
    # 请替换为你实际的权重路径
    TEST_WEIGHTS = "D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    if os.path.exists(TEST_WEIGHTS):
        model = MyDinoV3Model(
            weights_path=TEST_WEIGHTS,
            num_classes=10,
            model_name='base'
        )
        print("模型加载成功！")

        inputs = torch.randn(2, 3, 32, 32)
        logits = model(inputs)
        print(f"输入 (32x32) -> 输出 Logits: {logits.shape}")
    else:
        print(f"请修改脚本中的 TEST_WEIGHTS 路径进行测试")
