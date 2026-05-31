import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPyramidHeadStrict(nn.Module):
    """
    严格的频率金字塔，返回 4 种不同的空间分辨率:
      - level0: (B, out_ch, H0, W0)       (原始分辨率)
      - level1: (B, out_ch, H0//2, W0//2)
      - level2: (B, out_ch, H0//4, W0//4)
      - level3: (B, out_ch, H0//8, W0//8)
    同时返回从多尺度融合中池化得到的 freq_global (B, out_ch)，
    以及在使用时调整为每个请求对齐大小的相位掩码 P。
    """
    def __init__(self, out_ch: int = 128, base_channels: int = 64):
        super().__init__()
        self.out_ch = out_ch
        pc = base_channels

        # 处理每个尺度对数幅度的小型卷积
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, pc, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pc, pc, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, pc, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pc, pc, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, pc, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pc, pc, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, pc, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pc, pc, 3, padding=1), nn.ReLU(inplace=True)
        )

        # 将每个 pc 投影 -> out_ch
        self.proj0 = nn.Conv2d(pc, out_ch, 1)
        self.proj1 = nn.Conv2d(pc, out_ch, 1)
        self.proj2 = nn.Conv2d(pc, out_ch, 1)
        self.proj3 = nn.Conv2d(pc, out_ch, 1)

        # 精炼卷积 (可选)
        self.refine0 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.refine1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.refine2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.refine3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # 相位掩码卷积 (映射 |Δφ| -> P)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 全局融合 MLP
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch)
        )

    @staticmethod
    def _minimal_angle_diff(a, b):
        diff = a - b
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        return diff.abs()

    def _phase_diff_map(self, S: torch.Tensor):
        phi = torch.angle(S)
        if phi.dim() == 3:
            phi = phi.unsqueeze(1)
        phi_left = F.pad(phi, (1, 0, 0, 0))[:, :, :, :-1]
        phi_right = F.pad(phi, (0, 1, 0, 0))[:, :, :, 1:]
        phi_up = F.pad(phi, (0, 0, 1, 0))[:, :, :-1, :]
        phi_down = F.pad(phi, (0, 0, 0, 1))[:, :, 1:, :]

        dl = self._minimal_angle_diff(phi, phi_left)
        dr = self._minimal_angle_diff(phi, phi_right)
        du = self._minimal_angle_diff(phi, phi_up)
        dd = self._minimal_angle_diff(phi, phi_down)
        delta = (dl + dr + du + dd) / 4.0
        return delta  # (B,1,H0,W0)

    def forward(self, x_gray: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_gray: (B,1,H0,W0)
        returns:
          freq_pyr: 列表 [L0 (H0xW0), L1 (H0/2), L2 (H0/4), L3 (H0/8)]
          freq_global: (B, out_ch)
          P_full: (B,1,H0,W0)  -- 全分辨率掩码 (稍后可以进行插值)
          mag_log: (B,1,H0,W0)
        """
        B, C, H0, W0 = x_gray.shape
        # 快速傅里叶变换 (fft)
        S = torch.fft.fft2(x_gray)
        S = torch.fft.fftshift(S, dim=(-2, -1))
        mag = torch.abs(S)
        mag_log = torch.log1p(mag)

        # 生成 level0: 原始分辨率
        b0 = self.conv0(mag_log)          # (B,pc,H0,W0)
        l0 = self.proj0(b0)
        l0 = self.refine0(l0)

        # level1: pool2 (H0/2)
        mag1 = F.avg_pool2d(mag_log, kernel_size=2, stride=2, ceil_mode=True)
        b1 = self.conv1(mag1)
        l1 = self.proj1(b1)
        l1 = self.refine1(l1)            # (B,out_ch,H0/2,W0/2)

        # level2: pool4 (H0/4)
        mag2 = F.avg_pool2d(mag_log, kernel_size=4, stride=4, ceil_mode=True)
        b2 = self.conv2(mag2)
        l2 = self.proj2(b2)
        l2 = self.refine2(l2)

        # level3: pool8 (H0/8) (保护最小尺寸为 1)
        k = 8
        if min(H0, W0) // k < 1:
            k = max(1, min(H0, W0))
        mag3 = F.avg_pool2d(mag_log, kernel_size=k, stride=k, ceil_mode=True)
        b3 = self.conv3(mag3)
        l3 = self.proj3(b3)
        l3 = self.refine3(l3)

        # 在全分辨率下计算相位掩码
        delta_phi = self._phase_diff_map(S)  # (B,1,H0,W0)
        P_full = self.mask_conv(delta_phi)   # (B,1,H0,W0)

        # 全局嵌入: 池化 l0 (您也可以池化各层的组合)
        freq_global = self.global_mlp(l3)

        freq_pyr = [l0, l1, l2, l3]
        return freq_pyr, freq_global, P_full, mag_log
