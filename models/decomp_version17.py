import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class DyWAN_Pro(nn.Module):
    def __init__(self, channel, filter_length=8, hidden_dim=32, reg_factor=0.01):
        super().__init__()
        self.filter_length = filter_length
        self.reg_factor = reg_factor

        # 多尺度统计特征提取
        self.stat_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channel, hidden_dim),
            nn.GELU()
        )
        # 小波生成器
        self.wavelet_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, filter_length * 2)
        )

    def forward(self, x):
        # x: [B, C, L]
        stat = self.stat_net(x)                              # [B, hidden]
        f = self.wavelet_generator(stat)                     # [B, 2*filter_length]
        lo_f, hi_f = f.split(self.filter_length, dim=1)      # 各 [B, filter_length]

        # —— 这里指定为 float32，避免类型不匹配 ——
        kernel = torch.tensor([[-1, 1]], dtype=torch.float32, device=x.device).view(1,1,2)
        lo_smooth = F.conv1d(lo_f.unsqueeze(1), kernel, padding=1).abs().mean()

        ortho = self._orthogonality_constraint(lo_f, hi_f)
        return lo_f, hi_f, ortho + 0.1 * lo_smooth

    def _orthogonality_constraint(self, lo, hi):
        lo_n = lo / (lo.norm(dim=1, keepdim=True) + 1e-8)
        hi_n = hi / (hi.norm(dim=1, keepdim=True) + 1e-8)
        shift_loss = sum(
            (lo_n.unsqueeze(2) @ torch.roll(lo_n.unsqueeze(1), s, dims=2))
             .abs().mean()
            for s in range(1, 4)
        )
        amp_loss = (lo_n.pow(2).sum(1) - 1).abs().mean()
        return self.reg_factor * (shift_loss + amp_loss)


class WaveletODE_Pro(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, 3, padding=1, padding_mode='replicate')
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, in_channels, 3, padding=1, padding_mode='replicate')
        self.tanh  = nn.Tanh()
        self.damp  = nn.Parameter(torch.tensor(0.01))

    def forward(self, t, x):
        # x: [B, C, L]
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        h = self.tanh(h)
        return h - self.damp * x


class CrossScaleFusion(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.Sigmoid()
            )
            for _ in range(levels)
        ])

        # 保证 mid_channels >= 1
        mid_channels = max(1, channels // 4)
        self.atts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, mid_channels, 1),
                nn.GELU(),
                nn.Conv1d(mid_channels, channels, 1),
                nn.Sigmoid()
            )
            for _ in range(levels - 1)
        ])

    def forward(self, yl, yh_list):
        enhanced = []
        current = yl
        # 从最深层开始反向融合
        for i in range(len(yh_list) - 1, -1, -1):
            detail = yh_list[i]
            if i < len(self.atts):
                detail = detail * self.atts[i](current) + detail
            fused = current + self.gates[i](current) * detail
            current = fused
            enhanced.append(detail)
        enhanced.reverse()
        return current, enhanced


class AdvancedWaveletDecomp(nn.Module):
    def __init__(self, channel, level=3, time_points=8,
                 filter_length=8, dim=64, ode_solver='dopri5', ode_step=0.1):
        super().__init__()
        self.channel = channel
        self.level   = level
        self.filter_length = filter_length

        # 预处理 ODE
        self.pre_ode = WaveletODE_Pro(in_channels=channel, hidden_dim=dim)
        # 动态小波生成器
        self.dywan   = DyWAN_Pro(channel, filter_length, hidden_dim=dim)
        # 跨尺度融合
        self.csf     = CrossScaleFusion(channel, level)

        # 时间点
        self.train_t = torch.sort(torch.rand(time_points))[0]
        self.eval_t  = torch.linspace(0, 1, time_points)
        self.ode_solver = ode_solver
        self.ode_step   = ode_step

    def forward(self, x):
        B, C, L = x.shape
        # —— 1) 先 ODE 演化 ——
        t_pts = self.train_t if self.training else self.eval_t
        x_evolved = odeint(
            self.pre_ode,
            x,
            t_pts.to(x.device),
            method=self.ode_solver,
            options={'step_size': self.ode_step}
        )[-1]  # [B, C, L]

        # —— 2) 多级动态小波分解 ——
        approx   = x_evolved
        coeffs   = [approx]
        filters  = []
        ortho_ls = 0.0

        # 非对称填充参数
        pad_left  = self.filter_length // 2
        pad_right = self.filter_length - pad_left - 1

        for _ in range(self.level):
            lo, hi, ortho = self.dywan(approx)
            filters.append((lo, hi))
            ortho_ls += ortho

            ap = F.pad(approx, (pad_left, pad_right), mode='replicate')  # [B,C,L+FL-1]
            # 构造 per-sample per-channel 卷积核
            lo_k = lo.unsqueeze(1).repeat(1, C, 1)  # [B,C,FL]
            hi_k = hi.unsqueeze(1).repeat(1, C, 1)
            # 分组卷积，输出直接就是长度 L
            new_approx = F.conv1d(
                ap.view(1, B*C, -1),
                lo_k.view(B*C, 1, -1),
                groups=B*C
            ).view(B, C, L)
            detail = F.conv1d(
                ap.view(1, B*C, -1),
                hi_k.view(B*C, 1, -1),
                groups=B*C
            ).view(B, C, L)

            coeffs.append(detail)
            approx = new_approx

        # —— 3) 跨尺度融合 ——
        yl, enhanced = self.csf(coeffs[0], coeffs[1:])
        losses = {'ortho_loss': ortho_ls}

        return yl, enhanced, losses, filters

    def reconstruct(self, yl, yh_list, filters):
        B, C, L = yl.shape
        # 按顺序取出各层系数
        approx = yl
        details = yh_list

        pad_left  = self.filter_length // 2
        pad_right = self.filter_length - pad_left - 1

        # 从深层向浅层重构
        recon = approx
        for i in range(self.level-1, -1, -1):
            lo, hi = filters[i]
            det = details[i]

            rec_pad = F.pad(recon, (pad_left, pad_right), mode='replicate')
            det_pad = F.pad(det,   (pad_left, pad_right), mode='replicate')

            lo_k = lo.unsqueeze(1).repeat(1, C, 1)
            hi_k = hi.unsqueeze(1).repeat(1, C, 1)

            recon_lo = F.conv_transpose1d(
                rec_pad.view(1, B*C, -1),
                lo_k.view(B*C, 1, -1),
                groups=B*C
            ).view(B, C, -1)[..., pad_left:pad_left+L]

            recon_hi = F.conv_transpose1d(
                det_pad.view(1, B*C, -1),
                hi_k.view(B*C, 1, -1),
                groups=B*C
            ).view(B, C, -1)[..., pad_left:pad_left+L]

            recon = recon_lo + recon_hi

        return recon


# ====================== 测试代码 ======================
if __name__ == "__main__":
    batch_size = 4
    channels   = 3
    seq_len    = 128
    levels     = 3

    model = AdvancedWaveletDecomp(channel=channels, level=levels, dim=64)
    x = torch.randn(batch_size, channels, seq_len)

    yl, yh, losses, filters = model(x)
    print("输入:", x.shape)
    print("YL:", yl.shape)
    for i, d in enumerate(yh):
        print(f"YH[{i}]:", d.shape)

    recon = model.reconstruct(yl, yh, filters)
    print("重建:", recon.shape)
    print("MSE:", F.mse_loss(x, recon).item())
    print("正交损失:", losses['ortho_loss'].item())
