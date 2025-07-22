import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


def _upsample(x, target_len):
    B, L, C = x.shape
    if L == target_len:
        return x
    y = x.permute(0, 2, 1)
    y = F.interpolate(y, size=target_len, mode='linear', align_corners=False)
    return y.permute(0, 2, 1)


class MultiHeadDecomp(nn.Module):
    def __init__(self, nvar=7, hidden_dim=64, levels=3, device='cuda'):
        super().__init__()
        self.nvar = nvar
        self.hidden_dim = hidden_dim
        self.levels = levels
        self.device = device

        self.heads = nn.ModuleList([
            WaveletHead(nvar, levels),
            FourierHead(nvar, levels),
            TrendHead(nvar, levels),
            StochasticHead(nvar, levels)
        ])
        self.n_heads = len(self.heads)

        fusion_dim = self.n_heads * (levels + 1) * nvar
        self.query = nn.Linear(fusion_dim, hidden_dim)
        self.key = nn.Linear(fusion_dim, hidden_dim)
        self.value = nn.Linear(fusion_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, (levels + 1) * nvar)

        self.affine_w = nn.Parameter(torch.ones(levels + 1, nvar))
        self.affine_b = nn.Parameter(torch.zeros(levels + 1, nvar))

        self.head_weights = nn.Parameter(torch.ones(self.n_heads))

    def forward(self, x):
        B, T, C = x.shape
        self._T = T  # save original length
        raw = [head(x) for head in self.heads]
        comps = []
        for low, highs in raw:
            low_up = _upsample(low, T)
            highs_up = [_upsample(h, T) for h in highs]
            comps.append([low_up] + highs_up)

        tensor = torch.stack([torch.stack(h, dim=2) for h in comps], dim=2)
        B, T, H, L1, C = tensor.shape
        flat = tensor.view(B, T, H * L1 * C)

        Q = self.query(flat)
        K = self.key(flat)
        V = self.value(flat)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        W = F.softmax(scores, dim=-1)
        fused = torch.bmm(W, V)
        out = self.attn_out(fused).view(B, T, L1, C)

        lows = out[:, :, 0, :]
        highs = [out[:, :, i + 1, :] for i in range(self.levels)]

        lows = lows * self.affine_w[0] + self.affine_b[0]
        highs = [h * self.affine_w[i + 1] + self.affine_b[i + 1]
                 for i, h in enumerate(highs)]

        self._raw = raw
        return lows, highs

    def inverse(self, low, highs):
        recs = []
        for (l_raw, h_raw), head in zip(self._raw, self.heads):
            rec = head.inverse(l_raw, h_raw)
            # upsample rec to original length
            rec_up = _upsample(rec, self._T)
            recs.append(rec_up)
        stack = torch.stack(recs, dim=-1)
        w = F.softmax(self.head_weights, dim=0)
        recon = (stack * w).sum(dim=-1)
        return recon


class WaveletHead(nn.Module):
    def __init__(self, nvar, levels=3):
        super().__init__()
        self.levels = levels
        self.conv = nn.Conv1d(nvar, nvar, kernel_size=5, padding=2, groups=nvar)

    def forward(self, x):
        lows = x
        highs = []
        for _ in range(self.levels):
            smooth = self.conv(lows.permute(0, 2, 1)).permute(0, 2, 1)
            high = lows - smooth
            lows = smooth[:, ::2, :]
            highs.append(high[:, ::2, :])
        return lows, highs

    def inverse(self, low, highs):
        recon = low
        for high in reversed(highs):
            recon = _upsample(recon, high.shape[1])
            recon = recon + high
        return recon


class FourierHead(nn.Module):
    def __init__(self, nvar, levels=3, keep_ratio=0.2):
        super().__init__()
        self.levels = levels
        self.keep_ratio = keep_ratio

    def forward(self, x):
        low, high = self._decompose(x)
        highs = [high] + [torch.zeros_like(high) for _ in range(self.levels - 1)]
        return low, highs

    def _decompose(self, x):
        xf = fft.rfft(x, dim=1)
        cut = int(xf.shape[1] * self.keep_ratio)
        lf, hf = xf.clone(), xf.clone()
        lf[:, cut:] = 0; hf[:, :cut] = 0
        low = fft.irfft(lf, dim=1, n=x.shape[1])
        high = fft.irfft(hf, dim=1, n=x.shape[1])
        return low, high

    def inverse(self, low, highs):
        recon = low
        for h in highs:
            recon = recon + h
        return recon


class TrendHead(nn.Module):
    def __init__(self, nvar, levels=3, poly_degree=2):
        super().__init__()
        self.levels = levels
        self.poly_degree = poly_degree

    def forward(self, x):
        B, T, C = x.shape
        t = torch.linspace(0, 1, T, device=x.device).unsqueeze(-1)
        Tmat = torch.cat([t**i for i in range(self.poly_degree+1)], dim=1)
        lows, highs = [], []
        for i in range(C):
            Xi = x[:, :, i]
            A = Tmat.unsqueeze(0).expand(B, T, -1)
            sol = torch.linalg.lstsq(A, Xi.unsqueeze(-1)).solution
            trend = (A @ sol).squeeze(-1)
            residual = Xi - trend
            lows.append(trend.unsqueeze(-1)); highs.append(residual.unsqueeze(-1))
        low = torch.cat(lows, dim=-1)
        res = torch.cat(highs, dim=-1)
        highs = [res] + [torch.zeros_like(res) for _ in range(self.levels-1)]
        return low, highs

    def inverse(self, low, highs):
        return low + highs[0]


class StochasticHead(nn.Module):
    def __init__(self, nvar, levels=3, hidden_dim=32):
        super().__init__()
        self.levels = levels
        self.encoder = nn.GRU(nvar, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, nvar)

    def forward(self, x):
        _, h = self.encoder(x)
        B, T, _ = x.shape
        dec_in = torch.zeros(B, T, h.size(-1), device=x.device)
        dec, _ = self.decoder(dec_in, h)
        recon = self.out(dec)
        stochastic = x - recon
        highs = [stochastic] + [torch.zeros_like(stochastic) for _ in range(self.levels-1)]
        return recon, highs

    def inverse(self, low, highs):
        return low


# example usage
if __name__ == '__main__':
    model = MultiHeadDecomp(nvar=7, levels=3).to('cuda')
    x = torch.randn(32, 96, 7).to('cuda')
    low, highs = model(x)
    recon = model.inverse(low, highs)
    print('recon error:', torch.abs(recon - x).mean().item())
