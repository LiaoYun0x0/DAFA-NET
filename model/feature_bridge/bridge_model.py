import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import TransformerEncoderLayer # 实际上没有用到，可省略


class CrossAttentionBlock(nn.Module):
    # 此 Block 结构是标准的 Post-Norm Transformer Block，已经具有高可靠性。
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 前馈网络 (FFN) - 使用了 GELU 激活
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), # <--- 激活函数
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 1. Attention 路径 (LayerNorm + Attention + Residual)
        q_normalized = self.norm1(q)
        attn_output, _ = self.attn(query=q_normalized, key=k, value=v)
        attn_output = attn_output + q # 残差连接

        # 2. FFN 路径 (LayerNorm + FFN + Residual)
        norm_attn_output = self.norm2(attn_output)
        mlp_output = self.mlp(norm_attn_output)
        output = mlp_output + attn_output # 残差连接

        return output


class NetBridge(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super(NetBridge, self).__init__()

        # --- 1. EfficientNet 特征转换模块 (已包含 BatchNorm 和 ReLU) ---
        self.eff_48_conv = nn.Sequential(
            nn.Conv2d(56, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.eff_48_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.eff_24_conv = nn.Sequential(
            nn.Conv2d(160, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.eff_24_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.eff_12_conv = nn.Sequential(
            nn.Conv2d(448, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # --- 2. CLIP 特征转换模块 (增加 GELU 激活) ---
        self.clip_conv_1x1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU() # <--- 增强：在 CLIP 投影后使用 GELU
        )

        # --- 3. 融合后的输出模块（通道降维 2*D -> D, 已包含 BatchNorm 和 ReLU） ---
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # --- 4. Transformer 模块 ---
        self.block_1_2 = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.block_result_3 = CrossAttentionBlock(embed_dim, num_heads, dropout)

        # --- 5. 最终投影层 (增加 GELU 激活) ---
        self.final_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),  # LayerNorm(1024)
            nn.GELU(),                # <--- 增强：在最终降维前使用 GELU
            nn.Linear(embed_dim, 768)
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """ (N, H, W, D) -> (N, L, D) """
        N, H, W, D = x.shape
        return x.reshape(N, H * W, D)

    def forward(self, eff_feat_48, eff_feat_24, eff_feat_12,
                clip_feat_1, clip_feat_2, clip_feat_3):
        N = eff_feat_48.shape[0]

        # --- A/B. 特征转换（略） ---
        eff_48_target = self.eff_48_pool(self.eff_48_conv(eff_feat_48))
        eff_24_target = self.eff_24_pool(self.eff_24_conv(eff_feat_24))
        eff_12_proj = self.eff_12_conv(eff_feat_12)
        eff_12_target = F.interpolate(eff_12_proj, size=(16, 16), mode='bilinear', align_corners=False)

        clip_feat_1 = clip_feat_1.float()
        clip_feat_2 = clip_feat_2.float()
        clip_feat_3 = clip_feat_3.float()

        clip_1_proj = self.clip_conv_1x1(clip_feat_1)
        clip_1_target = clip_1_proj.permute(0, 2, 1).reshape(N, 1024, 16, 16)
        clip_2_proj = self.clip_conv_1x1(clip_feat_2)
        clip_2_target = clip_2_proj.permute(0, 2, 1).reshape(N, 1024, 16, 16)
        clip_3_proj = self.clip_conv_1x1(clip_feat_3)
        clip_3_target = clip_3_proj.permute(0, 2, 1).reshape(N, 1024, 16, 16)

        # --- C. 融合和转置 ---
        fused_1 = torch.cat([eff_48_target, clip_1_target], dim=1)
        output_1 = self.fusion_conv1(fused_1).permute(0, 2, 3, 1)

        fused_2 = torch.cat([eff_24_target, clip_2_target], dim=1)
        output_2 = self.fusion_conv2(fused_2).permute(0, 2, 3, 1)

        fused_3 = torch.cat([eff_12_target, clip_3_target], dim=1)
        output_3 = self.fusion_conv3(fused_3).permute(0, 2, 3, 1)

        # --- D. Transformer 融合 ---
        q1 = self._prepare_input(output_1)
        k1_v1 = self._prepare_input(output_2)
        q2_k2_v2 = self._prepare_input(output_3)

        result_1 = self.block_1_2(q=q1, k=k1_v1, v=k1_v1)
        final_result_seq = self.block_result_3(q=result_1, k=q2_k2_v2, v=q2_k2_v2)

        # --- E. 最终降维 (序列平均池化 + 线性投影) ---
        flattened_feat = final_result_seq.mean(dim=1)  # (N, 1024)
        final_output = self.final_projection(flattened_feat)  # (N, 768)

        return final_output
