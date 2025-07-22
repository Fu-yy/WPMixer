import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def _make_vector(tensor: torch.Tensor, device: torch.device) -> nn.Parameter:
    """创建并返回可学习的向量参数，封装到指定设备上"""
    return nn.Parameter(tensor.to(device).clone(), requires_grad=True)


class LearnableWaveletDecomp(nn.Module):
    """
    可学习小波分解模块，支持正交性和消失矩约束。
    输入/输出均为 (batch, channels, seq_len)。
    """
    def __init__(self,
                 nvar: int = 7,
                 init_wave: str = 'db1',
                 level: int = 3,
                 ortho_reg_weight: float = 0.1,
                 seq_len: int = 96,
                 pred_len: int = 96,
                 batch_size: int = 32,
                 mode: str = 'zero'):
        super().__init__()
        assert mode == 'zero', "目前仅支持 zero padding 模式"
        self.nvar = nvar
        self.level = level
        self.ortho_reg_weight = ortho_reg_weight
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        # 初始化经典小波滤波器系数
        base = pywt.Wavelet(init_wave)
        dec_lo = torch.tensor(base.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(base.dec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(base.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(base.rec_hi, dtype=torch.float32)

        # 统一设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 可学习滤波器
        self.dec_lo = _make_vector(dec_lo, self.device)
        self.dec_hi = _make_vector(dec_hi, self.device)
        self.rec_lo = _make_vector(rec_lo, self.device)
        self.rec_hi = _make_vector(rec_hi, self.device)
        # 仿射参数：level+1 组，各 nvar 通道
        self.affine_w = nn.Parameter(torch.ones(level + 1, nvar, device=self.device))
        self.affine_b = nn.Parameter(torch.zeros(level + 1, nvar, device=self.device))



        self.input_w_dim =self._dummy_forward(length=self.seq_len,batch_size=self.batch_size,channel=nvar)
        self.output_w_dim =self._dummy_forward(length=self.pred_len,batch_size=self.batch_size,channel=nvar)


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, L)
        Returns:
            low: (B, C, L_low)
            highs: list of (B, C, L_high_j)
            ortho_loss: 标量
            orig_len: 原始序列长度 L，用于重构时裁剪
        """
        x = x.to(self.device)
        B, C, L = x.shape
        assert C == self.nvar, f"输入通道 {C} != 模块配置 {self.nvar}"

        k = self.dec_lo.numel()
        pad = k // 2
        lo_filt = self.dec_lo.view(1, 1, k).repeat(C, 1, 1)
        hi_filt = self.dec_hi.view(1, 1, k).repeat(C, 1, 1)

        out = x
        highs = []
        for j in range(self.level):
            low_j = F.conv1d(out, lo_filt, stride=2, padding=pad, groups=C)
            high_j = F.conv1d(out, hi_filt, stride=2, padding=pad, groups=C)
            wj = self.affine_w[j].view(1, C, 1)
            bj = self.affine_b[j].view(1, C, 1)
            highs.append(high_j * wj + bj)
            out = low_j

        low = out * self.affine_w[self.level].view(1, C, 1) + self.affine_b[self.level].view(1, C, 1)
        ortho_loss = self._orthogonality_constraint()
        return low, highs, ortho_loss, L
    def _dummy_forward(self, length, batch_size, channel):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        # x = torch.ones(batch_size, channel, length, device=device)
        x = torch.ones(batch_size, channel, length)
        low, highs, ortho_loss, L= self.forward(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]
        yl = low
        yh = highs
        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l
    def inverse(self, low: torch.Tensor, highs: list, orig_len: int = None) -> torch.Tensor:
        """逆变换：通过转置卷积重构序列，并裁剪至原始长度"""
        low = low.to(self.device)
        B, C, _ = low.shape
        assert C == self.nvar

        k = self.rec_lo.numel()
        pad = k // 2
        g0 = self.rec_lo.view(1, 1, k).repeat(C, 1, 1)
        g1 = self.rec_hi.view(1, 1, k).repeat(C, 1, 1)

        out = (low - self.affine_b[self.level].view(1, C, 1)) / (
            self.affine_w[self.level].view(1, C, 1) + 1e-6
        )
        for j in reversed(range(self.level)):
            high_j = highs[j].to(self.device)
            high_j = (high_j - self.affine_b[j].view(1, C, 1)) / (
                self.affine_w[j].view(1, C, 1) + 1e-6
            )
            lo_up = F.conv_transpose1d(out, g0, stride=2, padding=pad,
                                       output_padding=out.shape[-1] % 2, groups=C)
            hi_up = F.conv_transpose1d(high_j, g1, stride=2, padding=pad,
                                       output_padding=high_j.shape[-1] % 2, groups=C)
            if lo_up.shape[-1] != hi_up.shape[-1]:
                m = max(lo_up.shape[-1], hi_up.shape[-1])
                lo_up = F.pad(lo_up, (0, m - lo_up.shape[-1]))
                hi_up = F.pad(hi_up, (0, m - hi_up.shape[-1]))
            out = lo_up + hi_up

        if orig_len is not None:
            out = out[..., :orig_len]
        return out

    def _orthogonality_constraint(self) -> torch.Tensor:
        one = torch.tensor(1.0, device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        dp = torch.dot(self.dec_lo, self.rec_lo) + torch.dot(self.dec_hi, self.rec_hi)
        ortho = F.mse_loss(dp, one)
        idx = torch.arange(self.dec_hi.numel(), device=self.dec_hi.device, dtype=self.dec_hi.dtype)
        moment = sum(torch.abs(torch.sum(self.dec_hi * (idx ** k))) for k in (1, 2))
        return self.ortho_reg_weight * (ortho + moment)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    level = 3
    seq_len = 192
    batch_size = 32
    pred_len = 336
    c_in = 7
    model = LearnableWaveletDecomp(nvar=c_in, level=level,seq_len=seq_len,pred_len=pred_len,batch_size=batch_size).to(device)
    x = torch.randn(32, 7, 96, device=device)
    low, highs, ortho_loss, L = model(x)
    in_dim_list = model.input_w_dim
    out_dim_list =model.output_w_dim

    low = torch.randn(32,7,out_dim_list[0])
    highs = [torch.randn(32,7,out_dim_list[1]),torch.randn(32,7,out_dim_list[2]),torch.randn(32,7,out_dim_list[3])]
    recon = model.inverse(low, highs, orig_len=pred_len)
    print(f"输入: {x.shape}, 重构: {recon.shape}")
    print(f"重构误差: {torch.mean(torch.abs(x - recon)).item():.6f}")
    print(f"Ort loss: {ortho_loss.item():.6f}")
