import torch
import torch.nn as nn
import torch.nn.functional as F

# Scheme 1: Learnable Filterbank Decomposition (1D)
class LearnableWavelet1D(nn.Module):
    def __init__(self, nvar, kernel_size=8, levels=1):
        """
        Args:
            nvar: number of input channels (variables)
            kernel_size: length of the 1-D filter
            levels: number of decomposition levels
        """
        super().__init__()
        self.levels = levels
        # two banks: lowpass and highpass
        # depthwise convs: one filter per channel
        self.lowpass = nn.Conv1d(nvar, nvar, kernel_size,
                                 padding=(kernel_size//2), groups=nvar, bias=False)
        self.highpass = nn.Conv1d(nvar, nvar, kernel_size,
                                  padding=(kernel_size//2), groups=nvar, bias=False)
        # initialization: approximate Haar wavelet
        for c in range(nvar):
            # simple Haar: low = [1/sqrt2, 1/sqrt2], high=[1/sqrt2, -1/sqrt2]
            lp = torch.zeros(kernel_size)
            hp = torch.zeros(kernel_size)
            lp[0], lp[1] = 1.0/torch.sqrt(torch.tensor(2.0)), 1.0/torch.sqrt(torch.tensor(2.0))
            hp[0], hp[1] = 1.0/torch.sqrt(torch.tensor(2.0)), -1.0/torch.sqrt(torch.tensor(2.0))
            self.lowpass.weight.data[c,0] = lp
            self.highpass.weight.data[c,0] = hp

    def forward(self, x):
        """
        x: (B, nvar, L)
        returns low approximation + list of high details per level
        """
        highs = []
        out = x
        for _ in range(self.levels):
            low = self.lowpass(out)
            high = self.highpass(out)
            # downsample by 2
            out = low[..., ::2]
            highs.append(high[..., ::2])
        return out, highs

    def inverse(self, low, highs):
        """
        low: (B, nvar, L/2^levels)
        highs: list of length levels of shape (B,nvar,L/2^levels)
        """
        out = low
        for high in reversed(highs):
            # upsample
            out = F.interpolate(out, scale_factor=2, mode='linear', align_corners=False)
            # inverse conv: use conv transpose
            out_low = self.lowpass(out)
            out_high = self.highpass(out)
            out = out_low + out_high
        return out


# Scheme 2: Encoder-Decoder based Multiscale Decomposition
class ConvAutoDecomp1D(nn.Module):
    def __init__(self, nvar, hidden_dim=32, levels=2):
        super().__init__()
        self.levels = levels
        # Encoder: successive strided convs
        encs = []
        in_ch = nvar
        for i in range(levels):
            out_ch = hidden_dim * (2**i)
            encs.append(nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            encs.append(nn.ReLU())
            in_ch = out_ch
        self.encoder = nn.Sequential(*encs)
        # Decoder: mirror with conv transpose
        decs = []
        for i in reversed(range(levels)):
            out_ch = nvar if i==0 else hidden_dim * (2**(i-1))
            decs.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if i>0:
                decs.append(nn.ReLU())
            in_ch = out_ch
        self.decoder = nn.Sequential(*decs)

    def forward(self, x):
        """
        x: (B,nvar,L)
        returns multiscale features: final code + intermediate features
        """
        features = []
        out = x
        # collect features at each scale
        for layer in self.encoder:
            out = layer(out)
            if isinstance(layer, nn.Conv1d):
                features.append(out)
        code = out
        recon = self.decoder(code)
        return code, features, recon


# Scheme 3: Spectral Self-Attention Decomposition
class SpectralDecomp1D(nn.Module):
    def __init__(self, nvar, n_freqs=49, embed_dim=64, heads=4):
        """
        n_freqs for rfft on length-96 yields 49 freq bins
        """
        super().__init__()
        self.n_freqs = n_freqs
        self.embed = nn.Linear(nvar*2, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)
        self.reconstruct = nn.Linear(embed_dim, nvar*2)

    def forward(self, x):
        """
        x: (B,nvar,L)
        returns: reconstructed signal
        """
        B, C, L = x.shape
        # real FFT
        spec = torch.fft.rfft(x, dim=-1)  # (B,C,Lf)
        real = spec.real
        imag = spec.imag
        # reshape to seq of freq tokens: (B, Lf, C*2)
        freqs = torch.cat([real, imag], dim=1).permute(0,2,1)
        # embed
        token = self.embed(freqs)  # (B,Lf,embed)
        attn_out, _ = self.mha(token, token, token)
        out_tok = self.reconstruct(attn_out)  # (B,Lf,C*2)
        # back to complex
        out_tok = out_tok.permute(0,2,1)
        real_hat, imag_hat = out_tok.chunk(2, dim=1)
        spec_hat = torch.complex(real_hat, imag_hat)
        # inverse FFT
        x_hat = torch.fft.irfft(spec_hat, n=L, dim=-1)
        return x_hat


# Example usage
def main():
    B, L, C = 32, 96, 7
    x = torch.randn(B, C, L)

    # Scheme 1
    model1 = LearnableWavelet1D(nvar=C, kernel_size=4, levels=2)
    low, highs = model1(x)
    low = torch.randn(32, 7, 25)
    highs = [torch.randn(32, 7, 49), torch.randn(32, 7, 25)]
    pred_len = 192
    x1 = model1.inverse(low, highs)
    print('Scheme1:', x1.shape)

    # Scheme 2
    model2 = ConvAutoDecomp1D(nvar=C, hidden_dim=16, levels=3)
    code, feats, recon = model2(x)
    print('Scheme2 code:', code.shape, 'recon:', recon.shape)

    # Scheme 3
    model3 = SpectralDecomp1D(nvar=C, n_freqs=L//2+1, embed_dim=32, heads=4)
    x3 = model3(x)
    print('Scheme3:', x3.shape)

if __name__ == '__main__':
    main()
