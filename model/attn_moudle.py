from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFuseDual(nn.Module):
    """
    双向 Cross-Attention (交叉注意力):
      - space -> freq (空间查询关注频率的键/值)
      - freq_global -> space (全局频率查询关注空间的键/值)
    输出融合后的空间图，并可选地返回注意力权重。
    """
    def __init__(self, in_ch: int, freq_ch: int, attn_embed: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.freq_ch = freq_ch
        self.attn_embed = attn_embed
        self.num_heads = num_heads

        # 投影
        self.qs = nn.Conv2d(in_ch, attn_embed, 1)
        self.ks = nn.Conv2d(in_ch, attn_embed, 1)
        self.vs = nn.Conv2d(in_ch, attn_embed, 1)

        # self.qf = nn.Conv2d(freq_ch, attn_embed, 1)
        self.kf = nn.Conv2d(freq_ch, attn_embed, 1)
        self.vf = nn.Conv2d(freq_ch, attn_embed, 1)

        # 全局频率 -> 查询
        self.freq_to_q = nn.Linear(freq_ch, attn_embed)

        # 多头注意力 (MHA) 模块
        self.mha_sf = nn.MultiheadAttention(embed_dim=attn_embed, num_heads=num_heads, dropout=dropout)
        self.mha_fs = nn.MultiheadAttention(embed_dim=attn_embed, num_heads=num_heads, dropout=dropout)

        # 输出投影回空间通道
        self.out_proj = nn.Conv2d(attn_embed, in_ch, 1)

        # SE 风格的通道门控
        mid = max(8, in_ch // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch),
            nn.Sigmoid(),
        )

    def _mha_forward(self, mha: nn.MultiheadAttention, q, k, v):
        """
        q,k,v 的形状为 (L, B, E)
        返回 out (L, B, E) 和 attn_weights (B, L, S)
        """
        out, attn_weights = mha(q, k, v, need_weights=True)  # out: (L,B,E); attn_weights: (B, L, S)
        return out, attn_weights

    def forward(self, F_s: torch.Tensor, F_f: torch.Tensor, freq_global: torch.Tensor, P: Optional[torch.Tensor] = None):
        """
        F_s: (B, C, H, W)
        F_f: (B, d, Hf, Wf)  -- 如果需要，将在外部插值到 (H,W)
        freq_global: (B, D)
        P: 相位掩码 (B,1,H_full,W_full) 或已调整为 (H,W)
        """
        B, C, H, W = F_s.shape
        # 确保 F_f 具有相同的空间尺寸 (如果不同，必须在外部进行插值)
        if F_f.shape[2:] != (H, W):
            F_f = F.interpolate(F_f, size=(H, W), mode='bilinear', align_corners=False)
        # 投影 -> (B, E, H, W)
        qs = self.qs(F_s)
        ks = self.ks(F_s)
        vs = self.vs(F_s)

        # qf = self.qf(F_f)
        kf = self.kf(F_f)
        vf = self.vf(F_f)

        N = H * W
        # 重塑为 (L, B, E)，其中 L=N
        q_s_seq = qs.view(B, self.attn_embed, N).permute(2, 0, 1)  # (N, B, E)
        k_f_seq = kf.view(B, self.attn_embed, N).permute(2, 0, 1)
        v_f_seq = vf.view(B, self.attn_embed, N).permute(2, 0, 1)

        # 1) 空间 -> 频率: 来自空间的查询关注频率的键/值
        out_sf_seq, attn_sf = self._mha_forward(self.mha_sf, q_s_seq, k_f_seq, v_f_seq)  # out: (N,B,E)
        out_sf = out_sf_seq.permute(1, 2, 0).contiguous().view(B, self.attn_embed, H, W)  # (B,E,H,W)

        # 2) 全局频率 -> 空间: 使用全局频率作为单个查询 token
        q_fg = self.freq_to_q(freq_global)  # (B, E)
        # 序列长度为 1: (1,B,E)
        q_fg_seq = q_fg.unsqueeze(0)
        k_s_seq = ks.view(B, self.attn_embed, N).permute(2, 0, 1)  # (N,B,E)
        v_s_seq = vs.view(B, self.attn_embed, N).permute(2, 0, 1)
        out_fs_seq, attn_fs = self._mha_forward(self.mha_fs, q_fg_seq, k_s_seq, v_s_seq)  # out_fs_seq: (1,B,E)
        # 通过广播聚合向量将 out_fs 扩展到空间图
        out_fs_vec = out_fs_seq.squeeze(0)  # (B,E)
        out_fs = out_fs_vec.view(B, self.attn_embed, 1, 1).expand(-1, -1, H, W)  # (B,E,H,W)

        # 结合两个方向的输出 (相加或拼接后投影)
        out_comb = out_sf + out_fs  # (B,E,H,W)
        out_spatial = self.out_proj(out_comb)  # (B,C,H,W)

        # 如果提供了相位掩码则应用它 (假设已经插值到 HxW)
        if P is not None:
            if P.shape[2:] != (H, W):
                P = F.interpolate(P, size=(H, W), mode='bilinear', align_corners=False)
            out_spatial = out_spatial * P

        # 残差 + SE
        fused = out_spatial + F_s
        scale = self.se(fused).view(B, C, 1, 1)
        fused = fused * scale

        # 同时返回注意力权重用于调试 (可选)
        # attn_sf: (B, N, N) 按照 mha 实现返回 (B, L, S) 其中 L=N, S=N
        # attn_fs: (B, 1, N) 其中查询长度 = 1
        return fused, attn_sf, attn_fs
