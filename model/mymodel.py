# dino_freq_dualfusion.py
# v1 版本
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将此导入替换为您真正的主干类
from model.dino_backbone import MyDinoV3Model
from model.frequent_backbone import FrequencyPyramidHeadStrict
from model.attn_moudle import CrossAttentionFuseDual


class DINOFreqDualFusionClassifier(nn.Module):
    """
    构建以下内容的包装器:
     - MyDinoV3Model 主干 (使用 get_intermediate_layers 提取层索引)
     - FrequencyPyramidHeadStrict 生成 4 个层级
     - 为每对 (4对) 构建 CrossAttentionFuseDual
     - 来自池化融合特征的最终分类器
    """
    def __init__(self, weights_path: str, freq_ch: int = 128, attn_embed: int = 256, heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.backbone = MyDinoV3Model(weights_path=weights_path, num_classes=10, model_name='base')
        self.freq_head = FrequencyPyramidHeadStrict(out_ch=freq_ch)
        # 创建 4 个融合模块 (我们保留原型 in_ch=768，稍后会进行懒加载适配)
        self.fuse_modules = nn.ModuleList([
            CrossAttentionFuseDual(in_ch=768, freq_ch=freq_ch, attn_embed=attn_embed, num_heads=heads)
            for _ in range(4)
        ])
        # 最终分类器 (拼接池化后的通道)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        device = x.device
        B = x.size(0)

        # 1) 从 DINO 提取 4 个中间特征
        feature_maps = self.backbone.get_intermediate_layers(x, n=[2, 4, 7, 11], reshape=True, norm=True)
        if isinstance(feature_maps, tuple):
            feature_maps = list(feature_maps)
        # 假设 feature_maps = [F3, F5, F8, F12]
        # 每个 Fi: (B, C_i, H_i, W_i)
        # 我们将频率金字塔层级分别对齐到每个特征图

        # 2) 在输入分辨率下构建频率金字塔
        if x.size(1) == 3:
            x_gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).to(device)
        else:
            x_gray = x.to(device)
        freq_pyr, freq_global, P_full, maglog = self.freq_head(x_gray)
        # freq_pyr: [L0(H0), L1(H0/2), L2(H0/4), L3(H0/8)]

        # # 3) attn 融合 dino 特征和 freq 频域特征
        fused_maps = []
        attn_records = []
        for i, fm in enumerate(feature_maps):
            Bf, Cf, Hi, Wi = fm.shape
            # 选取对应的频率层级 (映射关系: 高频->浅层)
            # 您可能需要根据惯例反转映射顺序。
            # 我们将映射: fm0(较浅) -> freq_pyr[0] (最高分辨率), fm1->freq_pyr[1], ...
            freq_map = freq_pyr[i]
            # 如果频率图空间尺寸 != 特征图空间尺寸，则进行插值
            if freq_map.shape[2:] != (Hi, Wi):
                freq_map_aligned = F.interpolate(freq_map, size=(Hi, Wi), mode='bilinear', align_corners=False)
            else:
                freq_map_aligned = freq_map
            # 如果通道不匹配，则懒加载适配融合模块
            if self.fuse_modules[i].in_ch != Cf:
                # 替换为具有匹配 in_ch 的模块
                self.fuse_modules[i] = CrossAttentionFuseDual(in_ch=Cf, freq_ch=freq_map_aligned.shape[1],
                                                              attn_embed=self.fuse_modules[i].attn_embed,
                                                              num_heads=self.fuse_modules[i].num_heads).to(device)
            # 同时为该分辨率准备掩码 P
            # 缩放（调整大小）到与当前正在处理的特征图 fm 相同的尺寸 (Hi, Wi)
            P_i = F.interpolate(P_full, size=(Hi, Wi), mode='bilinear', align_corners=False)
            fused, attn_sf, attn_fs = self.fuse_modules[i](fm, freq_map_aligned, freq_global, P_i)
            fused_maps.append(fused)
            attn_records.append((attn_sf, attn_fs))

        # 池化并拼接
        pooled = [F.adaptive_avg_pool2d(t, 1).view(B, -1) for t in fused_maps]  # each (B, C=768)
        cat = torch.cat(pooled, dim=1)
        logits = self.classifier(cat)

        # return logits, attn_records, freq_pyr, P_full, maglog
        return logits


# --------------------------
# 示例用法:
# --------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEST_WEIGHTS = "D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # 实例化您的主干 (确保它支持 get_intermediate_layers)
    model = DINOFreqDualFusionClassifier(weights_path=TEST_WEIGHTS, freq_ch=128, attn_embed=256, heads=4, num_classes=2).to(device)

    img = torch.randn(2, 3, 224, 224).to(device)
    logits, attn_records, freq_pyr, P_full, maglog = model(img)
    print("logits:", logits.shape)
    for i, f in enumerate(freq_pyr):
        print(f"freq level {i} shape:", f.shape)
    print("phase mask full shape:", P_full.shape)
