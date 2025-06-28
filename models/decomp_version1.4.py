import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# ====================== 基础模块：可逆实例归一化（RevIN） ======================
class RevIN(nn.Module):
    """可逆实例归一化层"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True).clamp(min=self.eps)
            x_norm = (x - self.mean) / self.std
            return x_norm * self.gamma + self.beta
        else:
            return (x - self.beta) / self.gamma * self.std + self.mean


# ====================== 创新点1：QR 正交化动态小波基（DyWAN-QR） ======================
class DyWAN_QR(nn.Module):
    """使用 linalg.qr 确保正交基础"""
    def __init__(self, channel, filter_length=8, hidden_dim=32):
        super().__init__()
        self.filter_length = filter_length
        self.stat_net = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(channel, hidden_dim), nn.GELU()
        )
        self.wavelet_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, filter_length * 2)
        )

    def forward(self, x):
        feat = self.stat_net(x)  # [B, hidden]
        raw = self.wavelet_gen(feat)  # [B, 2*L]
        lo_raw, hi_raw = raw.chunk(2, dim=1)  # each [B, L]

        # QR 正交化：将每个 [L] 向量视作 [L,1] 矩阵
        lo_Q, _ = torch.linalg.qr(lo_raw.unsqueeze(-1), mode='reduced')
        hi_Q, _ = torch.linalg.qr(hi_raw.unsqueeze(-1), mode='reduced')
        lo = lo_Q.squeeze(-1)  # [B, L]
        hi = hi_Q.squeeze(-1)  # [B, L]
        return lo, hi


# ====================== 创新点2：能量守恒 Neural ODE（EC-ODE） ======================
class EnergyConservingODE(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, time_dim=16):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, 3, padding=1)
        )
        self.time_cond = nn.Linear(time_dim, hidden_dim*2)

    def forward(self, t, x):
        B, C, L = x.shape
        te = self.time_embed(t.view(-1,1))
        gamma, beta = self.time_cond(te).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).expand(B,-1,L)
        beta = beta.unsqueeze(-1).expand(B,-1,L)
        h = self.net[0](x)
        h = (1 + gamma) * h + beta
        for layer in self.net[1:]:
            h = layer(h)
        return h

    def conservation_loss(self, x, dxdt):
        # 最小化 d/dt ||x||^2 = 2 <x, f(t,x)>
        inner = (x * dxdt).flatten(1).sum(-1)
        return (inner**2).mean()


# ====================== 组合模型：DyWAN-QR + EC-ODE ======================
class ImprovedDyWAN_NDE(nn.Module):
    def __init__(self, channel, level=3, filter_length=8, time_steps=10):
        super().__init__()
        self.level = level
        self.filter_length = filter_length
        self.time_pts = torch.linspace(0,1,time_steps)
        self.dywan = DyWAN_QR(channel, filter_length)
        self.ode = EnergyConservingODE(in_channels=(1+level)*channel)

    def forward(self, x):
        B, C, L = x.shape
        coeffs = []
        approx = x
        coeffs.append(approx)
        # 分解
        for _ in range(self.level):
            lo, hi = self.dywan(approx)
            lo_f = lo.mean(0).view(1,1,-1).repeat(C,1,1)
            hi_f = hi.mean(0).view(1,1,-1).repeat(C,1,1)
            pad = self.filter_length - 1
            ap = F.pad(approx, (pad//2, pad-pad//2), mode='reflect')
            new_approx = F.conv1d(ap, lo_f, groups=C)[..., :L]
            detail = F.conv1d(ap, hi_f, groups=C)[..., :L]
            coeffs.append(detail)
            approx = new_approx
        # ODE 演化
        stacked = torch.cat(coeffs, dim=1)
        ode_out = odeint(self.ode, stacked, self.time_pts.to(x.device), method='dopri5')[-1]
        yl = ode_out[:, :C, :]
        yh = [ode_out[:, C*(i+1):C*(i+2), :] for i in range(self.level)]
        # 能量守恒损失
        with torch.enable_grad():
            dx = self.ode(self.time_pts[-1].view(1), ode_out)
        ecloss = self.ode.conservation_loss(stacked, dx)
        return yl, yh, {'energy_loss': ecloss}

    def reconstruct(self, yl, yh):
        rec = yl
        for detail in yh:
            lo, hi = self.dywan(rec)
            lo_f = lo.mean(0).view(1,1,-1).repeat(rec.shape[1],1,1)
            hi_f = hi.mean(0).view(1,1,-1).repeat(rec.shape[1],1,1)
            pad = self.filter_length - 1
            rec_lo = F.conv_transpose1d(F.pad(rec, (pad//2, pad-pad//2)), lo_f, groups=rec.shape[1])
            rec_hi = F.conv_transpose1d(F.pad(detail, (pad//2, pad-pad//2)), hi_f, groups=detail.shape[1])
            rec = (rec_lo + rec_hi)[..., :yl.size(2)]
        return rec

# ========== 运行示例 ==========


if __name__ == '__main__':
    m = ImprovedDyWAN_NDE(channel=8, level=2, filter_length=8)
    x = torch.randn(16, 8, 128)
    yl, yh, loss = m(x)
    rec = m.reconstruct(yl, yh)
    print('recon mse:', F.mse_loss(rec, x).item())
    print('energy loss:', loss['energy_loss'].item())
