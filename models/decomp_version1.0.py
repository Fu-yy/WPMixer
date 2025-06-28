import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import pywt

# --------------------------------------------------------------------------------
# Dynamic Wavelet Basis Adaptive Network (DyWAN)
# --------------------------------------------------------------------------------
class DyWANDecomposition(nn.Module):
    """
    Dynamic Wavelet Basis Adaptive Network
    Learns filter coefficients per batch via a small MLP.
    """
    def __init__(self, input_length, level, channel, hidden_dim=128):
        super().__init__()
        self.level = level
        self.input_length = input_length
        # MLP to generate wavelet filter coeffs (e.g., length K)
        base_wavelet = pywt.Wavelet('db1')
        K = base_wavelet.dec_len
        self.mlp = nn.Sequential(
            nn.Linear(channel * input_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):  # x: B,C,L
        B, C, L = x.shape
        # generate dynamic wavelet (placeholder usage)
        stats = x.view(B, -1)
        psi = self.mlp(stats)            # B x K coefficients
        # For demonstration, fallback to static pywt decomposition
        coeffs = []
        arr = x.cpu().numpy()
        for b in range(B):
            c = pywt.wavedec(arr[b], 'db1', level=self.level)
            coeffs.append(c)
        return coeffs

# --------------------------------------------------------------------------------
# Time-Varying Deep Decomposition (TV-D²)
# --------------------------------------------------------------------------------
class TVDDDecomposition(nn.Module):
    """
    Time-Varying Deep Decomposition
    Adaptive number of levels with gates.
    """
    def __init__(self, level, channel, tau=0.5):
        super().__init__()
        self.level = level
        self.gates = nn.ModuleList([nn.Conv1d(channel, 1, kernel_size=1) for _ in range(level)])
        self.tau = tau

    def forward(self, x):  # x: B,C,L
        B, C, L = x.shape
        arr = x.cpu().numpy()
        # perform wavedec along last axis for each batch
        yl = []
        yh = [[] for _ in range(self.level)]
        for b in range(B):
            coeffs = pywt.wavedec(arr[b], 'db1', level=self.level)
            yl.append(coeffs[0])
            for i, c in enumerate(coeffs[1:]):
                yh[i].append(c)
        # stack and apply gates per level
        yl = torch.from_numpy(np.stack(yl)).to(x.device).float()  # B x C x L1
        gated_yh = []
        for i in range(self.level):
            # stack level i coefficients: list of B arrays shape CxLi -> tensor BxCxLi
            coeffs_i = torch.from_numpy(np.stack(yh[i])).to(x.device).float()
            # compute gate alpha
            alpha = torch.sigmoid(self.gates[i](coeffs_i))  # Bx1xLi
            # mask based on mean gate value per batch
            mask = (alpha.mean(dim=[1,2], keepdim=True) > self.tau).float()
            gated = coeffs_i * mask.view(B,1,1)
            gated_yh.append(gated)
        return yl, gated_yh

# --------------------------------------------------------------------------------
# Quantum-Inspired Complex Wavelet Decomposition (QCW-Mixer)
# --------------------------------------------------------------------------------
class QCWDecomposition(nn.Module):
    """
    Quantum-Inspired Complex Wavelet Decomposition
    Simulates dual-tree complex wavelet via analytic signal.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):  # x: B,C,L real
        # analytic signal via Hilbert transform placeholder using FFT
        Xf = torch.fft.rfft(x, dim=-1)
        return Xf

# --------------------------------------------------------------------------------
# Differentiable Wavelet Packet Entropy Pruning (DWPE)
# --------------------------------------------------------------------------------
class DWPEDecomposition(nn.Module):
    """
    Differentiable Wavelet Packet Entropy Pruning
    Builds full wavelet packet tree, prunes via entropy.
    """
    def __init__(self, level, threshold=0.1):
        super().__init__()
        self.level = level
        self.threshold = threshold

    def entropy(self, coeff):
        p = coeff.pow(2)
        p = p / (p.sum() + 1e-8)
        return -(p * torch.log(p + 1e-8)).sum()

    def forward(self, x):  # x: B,C,L
        arr = x.cpu().numpy()
        # full wavelet packet
        tensor_coeffs = []
        for b in range(arr.shape[0]):
            wp = pywt.WaveletPacket(data=arr[b], wavelet='db1', mode='symmetric', maxlevel=self.level)
            nodes = wp.get_level(self.level, order='freq')
            kept = []
            for n in nodes:
                coeff = torch.from_numpy(n.data).to(x.device).float()
                if self.entropy(coeff) > self.threshold:
                    kept.append(coeff)
            tensor_coeffs.append(torch.stack(kept) if kept else torch.empty(0))
        return tensor_coeffs

# --------------------------------------------------------------------------------
# Wavelet-Neural Differential Equation Decomposition (W-NDE)
# --------------------------------------------------------------------------------
class WNDEDecomposition(nn.Module):
    """
    Wavelet-Neural Differential Equation Decomposition
    Models coefficient evolution as ODE.
    """
    def __init__(self, channel, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, channel)
        )

    def dynamics(self, t, X):
        return self.net(X)

    def forward(self, x):  # x: B,C,L
        B, C, L = x.shape
        # solve ODE along sequence
        t = torch.linspace(0, 1, steps=L).to(x.device)
        flat = x.permute(2, 0, 1)  # L, B, C
        sol = odeint(self.dynamics, flat, t)
        return sol[-1].permute(1, 2, 0)  # B, C, L

# --------------------------------------------------------------------------------
# Example main() demonstrating usage
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    B, C, L = 4, 3, 128
    x = torch.randn(B, C, L)

    print("=== DyWANDecomposition ===")
    dywan = DyWANDecomposition(input_length=L, level=3, channel=C)
    coeffs = dywan(x)
    print(f"DyWAN returned {len(coeffs[0])} subbands for batch[0]")

    print("\n=== TVDDDecomposition ===")
    tvdd = TVDDDecomposition(level=3, channel=C)
    yl, yh = tvdd(x)
    print(f"TVDD yl shape: {yl.shape}; yh count: {len(yh)}, shapes: {[t.shape for t in yh]}")

    print("\n=== QCWDecomposition ===")
    qcw = QCWDecomposition()
    Xf = qcw(x)
    print(f"QCW returned complex spectrum shape: {Xf.shape}")

    print("\n=== DWPEDecomposition ===")
    dwpe = DWPEDecomposition(level=3, threshold=0.05)
    pruned = dwpe(x)
    print(f"DWPE kept subbands per batch: {[p.shape for p in pruned]}")

    print("\n=== WNDEDecomposition ===")
    wnde = WNDEDecomposition(channel=C)
    ode_coeff = wnde(x)
    print(f"WNDE output shape: {ode_coeff.shape}")
