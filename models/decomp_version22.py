import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasPadding(nn.Module):
    """Boundary anti-aliasing with depthwise lowpass filter"""

    def __init__(self, channels):
        super().__init__()
        self.lowpass = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(channels, channels, kernel_size=3, groups=channels),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.lowpass(x)


class NonSubsampledLifting(nn.Module):
    """Non-subsampled lifting: predict-update keeping time dimension"""

    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        pad = (dilation * (kernel_size - 1)) // 2
        self.P = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, groups=channels),
            nn.GELU()
        )
        self.U = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, groups=channels),
            nn.GELU()
        )
        self.anti_alias = AntiAliasPadding(channels)

    def forward(self, x):
        x_aa = self.anti_alias(x)
        p = self.P(x_aa)
        detail = x_aa - p
        u = self.U(detail)
        approx = (x_aa + u) * 0.5
        return approx, detail


class TemporalGate(nn.Module):
    """Temporal gating across all coefficients"""

    def __init__(self, channels, num_coeffs):
        super().__init__()
        in_ch = channels * num_coeffs
        self.conv = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_ch, in_ch * 2, kernel_size=3),
            nn.GELU(),
            nn.Conv1d(in_ch * 2, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, coeffs):
        stacked = torch.cat(coeffs, dim=1)
        gates = self.conv(stacked)
        splits = torch.split(gates, coeffs[0].shape[1], dim=1)
        return [c * g for c, g in zip(coeffs, splits)]


class MultiScalePredictor(nn.Module):
    """Fuse multi-scale features and predict future per variable"""

    def __init__(self, channels, pred_length, num_coeffs):
        super().__init__()
        in_ch = channels * num_coeffs
        self.pred_length = pred_length
        self.fusion = nn.Sequential(
            nn.Conv1d(in_ch, channels * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.GELU()
        )
        # output channels = channels * pred_length
        self.time_reduce = nn.Conv1d(channels * 2, channels * pred_length, kernel_size=1)

    def forward(self, coeffs):
        # align lengths
        min_l = min(c.shape[-1] for c in coeffs)
        aligned = [c[..., :min_l] for c in coeffs]
        fused = torch.cat(aligned, dim=1)
        x = self.fusion(fused)
        x = self.time_reduce(x)  # [B, channels*pred_length, min_l]
        # average time dimension
        x = x.mean(dim=-1)      # [B, channels*pred_length]
        # reshape to [B, channels, pred_length]
        B = x.shape[0]
        return x.view(B, -1, self.pred_length)


class UnifiedMultiLevel_new(nn.Module):
    """Unified non-subsampled lifting multi-scale predictor"""

    def __init__(self, channels=1, levels=3, kernel_size=5,
                 lambda_d=0.01, pred_length=24,batch_size=32,input_length=96,device='cpu'):
        super().__init__()
        self.lambda_d = lambda_d
        self.pred_length = pred_length
        self.device = device
        # decompose & recon blocks
        dilations = [2 ** i for i in range(levels)]
        self.decompose = nn.ModuleList([
            NonSubsampledLifting(channels, kernel_size, d)
            for d in dilations
        ])
        self.recon = nn.ModuleList([
            NonSubsampledLifting(channels, kernel_size, d)
            for d in reversed(dilations)
        ])
        num_coeffs = levels + 1
        self.temporal_gate = TemporalGate(channels, num_coeffs)

        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length,batch_size,channels,self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length,batch_size,channels,self.device)  # required length of the pred seq after decom



    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        coeffs, losses= self.decompose_func(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]
        yl = coeffs[-1]
        yh = coeffs[:-1]
        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l


    def decompose_func(self,x):
        coeffs = []
        approx = x
        reg_loss = torch.tensor(0., device=x.device)
        for blk in self.decompose:
            approx, detail = blk(approx)
            coeffs.append(detail)
            reg_loss = reg_loss + self.lambda_d * F.huber_loss(detail, torch.zeros_like(detail))
        coeffs.append(approx)
        gated = self.temporal_gate(coeffs)
        return gated,reg_loss

    def recon_func(self,gated):
        # reconstruction
        current = gated[-1]
        for idx, blk in enumerate(self.recon):
            detail = gated[-(idx + 2)]
            u_out = blk.U(detail)
            x_rec = 2 * current - u_out
            p_out = blk.P(x_rec)
            current = x_rec + p_out
        recon = current
        return recon
    def forward(self, x, future=None):
        # x: [B, channels, L], future: [B, channels, pred_length]
        # coeffs = []
        # approx = x
        # reg_loss = torch.tensor(0., device=x.device)
        # for blk in self.decompose:
        #     approx, detail = blk(approx)
        #     coeffs.append(detail)
        #     reg_loss = reg_loss + self.lambda_d * F.huber_loss(detail, torch.zeros_like(detail))
        # coeffs.append(approx)
        gated,reg_loss = self.decompose_func(x)
        # prediction: [B, channels, pred_length]

        # pred = self.predictor(gated)
        # reconstruction
        # current = gated[-1]
        # for idx, blk in enumerate(self.recon):
        #     detail = gated[-(idx+2)]
        #     u_out = blk.U(detail)
        #     x_rec = 2 * current - u_out
        #     p_out = blk.P(x_rec)
        #     current = x_rec + p_out
        # recon = current
        recon = self.recon_func(gated)
        # losses
        losses = {
            'reg_loss': reg_loss,
            'recon_loss': F.mse_loss(recon, x)
        }

        losses['total_loss'] = losses['recon_loss'] + reg_loss
        return { 'recon': recon, 'coeffs': gated, 'losses': losses}


# 示例测试
if __name__ == "__main__":
    B, C, L = 32, 7, 96
    pred_length = 96
    levels = 3
    model = UnifiedMultiLevel_new(channels=C, levels=levels, pred_length=pred_length)
    x = torch.randn(B, C, L)
    future = torch.randn(B, C, pred_length)
    out = model(x)
    print("pred:", out['prediction'].shape)   # [32, 7, 48]
    print("recon:", out['recon'].shape)       # [32, 7, L]
    print("coeffs len:", len(out['coeffs']))  # levels+1
