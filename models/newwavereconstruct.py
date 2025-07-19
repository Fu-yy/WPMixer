import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class Splitting(nn.Module):
    """
    信号分解模块：交替采样
    """
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        even = x[..., ::2]
        odd  = x[..., 1::2]
        return even, odd

class LiftingPredictor(nn.Module):
    """
    提升框架中的预测(P)和更新(U)模块
    """
    def __init__(self, channels: int, length: int, k_size: int = 4):
        super().__init__()
        pad = (k_size // 2, k_size - 1 - k_size // 2)
        self.P = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(channels, channels, k_size, groups=channels),
            nn.GELU(),
            nn.LayerNorm([channels, length])
        )
        self.U = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(channels, channels, k_size, groups=channels),
            nn.GELU(),
            nn.LayerNorm([channels, length])
        )

    def forward(self, even: torch.Tensor, odd: torch.Tensor):
        # even, odd: (B, C, L)
        c = even + self.U(odd)
        d = odd  - self.P(c)
        return c, d

class SharedWaveNet(nn.Module):
    """
    端到端共享小波基的多级分解-重构网络
    包含：
      - 可学习低/高通滤波器（analysis & synthesis共享）
      - 提升预测器(LiftingPredictor)
      - 严格参数化逆变换(ParametricIDWT)
    """
    def __init__(self,
                 init_wavelet: str,
                 levels: int,
                 in_channels: int,
                 length: int,
                 k_size: int = 4):
        super().__init__()
        # 裂分器
        self.splitter = Splitting()
        self.levels = levels
        self.in_channels = in_channels
        self.length = length

        # 初始化经典小波滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo, dtype=torch.float)
        dec_hi = torch.tensor(wavelet.dec_hi, dtype=torch.float)
        rec_lo = torch.tensor(wavelet.rec_lo, dtype=torch.float)
        rec_hi = torch.tensor(wavelet.rec_hi, dtype=torch.float)
        self.ksz = dec_lo.numel()

        # 可学习分析与合成滤波器，共享同一组参数
        self.lo_filters = nn.Parameter(dec_lo.view(1,1,-1).repeat(in_channels,1,1))
        self.hi_filters = nn.Parameter(dec_hi.view(1,1,-1).repeat(in_channels,1,1))
        self.rec_lo_filters = nn.Parameter(rec_lo.view(1,1,-1).repeat(in_channels,1,1))
        self.rec_hi_filters = nn.Parameter(rec_hi.view(1,1,-1).repeat(in_channels,1,1))

        # LiftingPredictor 列表
        self.lift_preds = nn.ModuleList()
        for lvl in range(levels):
            L_l = length // (2 ** (lvl+1))
            self.lift_preds.append(LiftingPredictor(in_channels, L_l, k_size))

    def forward(self, x: torch.Tensor):
        """
        输入 x: (B, C, L)
        输出:
          - recon: 重构信号 (B, C, L)
          - coeffs: 分解系数 { 'approxs': [a_L], 'details': [d_L,...,d_1] }
        """
        B, C, L = x.shape
        # 分解路径
        approx = x
        details = []
        for lvl in range(self.levels):
            # 裂分 & 卷积分析 (depthwise conv)
            even, odd = self.splitter(approx)
            # padding same
            total_pad = self.ksz - 1
            pad_l = total_pad // 2
            pad_r = total_pad - pad_l
            e_pad = F.pad(even, (pad_l, pad_r), mode='reflect')
            o_pad = F.pad(odd,  (pad_l, pad_r), mode='reflect')
            lo = F.conv1d(e_pad, self.lo_filters, groups=C)
            hi = F.conv1d(o_pad, self.hi_filters, groups=C)
            # 提升 refine
            c, d = self.lift_preds[lvl](lo, hi)
            approx = c
            details.append(d)
        # 最终近似
        a_L = approx

        # 重构路径（ParametricIDWT）
        current = a_L
        for lvl in reversed(range(self.levels)):
            d = details[lvl]
            # 上采样插零
            up_a = x.new_zeros((*current.shape[:2], current.shape[2]*2))
            up_d = x.new_zeros((*d.shape[:2], d.shape[2]*2))
            up_a[..., ::2] = current
            up_d[..., ::2] = d
            # same padding
            total_pad = self.ksz - 1
            pad_l = total_pad // 2
            pad_r = total_pad - pad_l
            pad_a = F.pad(up_a, (pad_l, pad_r), mode='reflect')
            pad_d = F.pad(up_d, (pad_l, pad_r), mode='reflect')
            # depthwise conv 重构
            lo = F.conv1d(pad_a, self.rec_lo_filters, groups=C)
            hi = F.conv1d(pad_d, self.rec_hi_filters, groups=C)
            current = lo + hi
        recon = current
        return recon, {'approxs': a_L, 'details': details}

    def wavelet_constraint_loss(self):
        """正交与归一约束，应用于 lo/hi 和 rec_lo/rec_hi"""
        loss = 0
        for f1, f2 in [(self.lo_filters, self.hi_filters), (self.rec_lo_filters, self.rec_hi_filters)]:
            # 每通道检查正交归一
            for i in range(self.in_channels):
                v1 = f1[i].view(-1)
                v2 = f2[i].view(-1)
                loss += torch.abs(torch.dot(v1, v2))
                loss += torch.abs(torch.norm(v1) - 1) + torch.abs(torch.norm(v2) - 1)
        return 1e-2 * loss

# 用法示例：
model = SharedWaveNet('db4', levels=3, in_channels=1, length=512)
x = torch.randn(4,1,512)
recon, coeffs = model(x)
rec_loss = F.mse_loss(recon, x)
orth_loss = model.wavelet_constraint_loss()
loss = rec_loss + orth_loss
