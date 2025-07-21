import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from models.wavelet_patch_mixer_back_20250628093416 import plot_tensors


class Splitting(nn.Module):
    """信号分割模块"""
    def __init__(self, channel_first=True):
        super().__init__()
        if channel_first:
            self.conv_even = lambda x: x[:, :, ::2]
            self.conv_odd  = lambda x: x[:, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, ::2, :]
            self.conv_odd  = lambda x: x[:, 1::2, :]

    def forward(self, x):
        return self.conv_even(x), self.conv_odd(x)


class LiftingPredictor(nn.Module):
    """自适应提升预测器"""
    def __init__(self, in_channels, length, k_size=4):
        super().__init__()
        pad = (k_size // 2, k_size - 1 - k_size // 2)  # (2,1) for k_size=4
        self.P = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_channels, in_channels, k_size, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, length])
        )
        self.U = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_channels, in_channels, k_size, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, length])
        )

    def forward(self, x_even, x_odd):
        c = x_even + self.U(x_odd)
        d = x_odd  - self.P(c)
        return c, d


class LiftBoostLearnWave(nn.Module):
    """
    提升增强型可学习小波分解（修正了 even-kernel 的 symmetric “same” padding）
    """
    def __init__(self, init_wavelet, num_filters,batch_size, levels, input_length,pred_length,channel,k_size=4, device='cpu'):
        super().__init__()
        self.levels      = levels
        self.num_filters = num_filters
        self.split       = Splitting(channel_first=True)
        self.channel = channel
        # 传统小波低/高通滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo, dec_hi = torch.tensor(wavelet.dec_lo).float(), torch.tensor(wavelet.dec_hi).float()
        self.device=device

        self.ksz = dec_lo.numel()  # 假设=8（偶数）

        # 可学习滤波器
        self.lo_filters     = nn.Parameter(dec_lo.repeat(num_filters, 1, 1), requires_grad=True).to(self.device)
        self.hi_filters     = nn.Parameter(dec_hi.repeat(num_filters, 1, 1), requires_grad=True).to(self.device)
        self.filter_weights = nn.Parameter(torch.ones(num_filters), requires_grad=True).to(self.device)

        # 重构
        rec_lo = torch.tensor(wavelet.rec_lo, dtype=torch.float)
        rec_hi = torch.tensor(wavelet.rec_hi, dtype=torch.float)

        self.rec_lo_filters = nn.Parameter(rec_lo.view(1,1,-1).repeat(num_filters,1,1),requires_grad=True)
        self.rec_hi_filters = nn.Parameter(rec_hi.view(1,1,-1).repeat(num_filters,1,1),requires_grad=True)
        self.rec_filter_weights = nn.Parameter(torch.ones(num_filters), requires_grad=True)



        # 每级的 lifting predictor
        self.lifting_predictors = nn.ModuleList()
        for i in range(levels):
            length_i = input_length // (2 ** (i + 1))
            self.lifting_predictors.append(LiftingPredictor(channel, length_i, k_size=k_size).to(self.device))

        # 归一化层
        self.norm_x = nn.InstanceNorm1d(1)
        self.norm_d = nn.InstanceNorm1d(1)
        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length,batch_size,channel,self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length,batch_size,channel,self.device)  # required length of the pred seq after decom

    def wavelet_constraint_loss(self):
        orth, energy = 0.0, 0.0
        for i in range(self.num_filters):
            lo = self.lo_filters[i].squeeze()
            hi = self.hi_filters[i].squeeze()
            orth   += torch.abs(torch.dot(lo, hi))
            energy += torch.abs(torch.norm(lo) - 1) + torch.abs(torch.norm(hi) - 1)
        return 0.01 * orth + 0.01 * energy

    def _get_transient_and_details(self, x):
        B, C, L = x.shape
        trans_feats, details = [], []
        current = x
        for lvl in range(self.levels):
            even, odd = self.split(current)
            c, d = self.lifting_predictors[lvl](even, odd)
            details.append(d)

            # 峭度：归一化后按通道平均
            mu    = d.mean(-1, keepdim=True)
            sigma = d.std(-1, keepdim=True, unbiased=False).clamp_min(1e-8)
            kurt  = ((d - mu)/sigma)**4
            kurt  = kurt.mean(dim=[1,2], keepdim=True) - 3  # (B,1,1)
            trans_feats.append(kurt.squeeze(-1).squeeze(-1))  # (B,)

            current = c

        trans_feats = torch.stack(trans_feats, dim=1)  # (B, levels)
        return trans_feats, details

    import torch.nn.functional as F

    def forward(self, x):
        # x: (B,1,L)
        B, C, L0 = x.shape
        trans_feats, details = self._get_transient_and_details(x)

        energy_feats = []
        current = x
        for lvl in range(self.levels):
            even, odd = self.split(current)
            fe = even.reshape(B*C, 1, -1)
            fo = odd .reshape(B*C, 1, -1)

            # —— 修正 “same” padding for even ksz ——
            pad_left  = self.ksz//2 - 1  # 8//2 -1 = 3
            pad_right = self.ksz//2      # 4
            # total_pad = self.ksz - 1
            # pad_left = total_pad // 2
            # pad_right = total_pad - pad_left

            fe_p = F.pad(fe,  (pad_left, pad_right), mode='reflect')
            fo_p = F.pad(fo,  (pad_left, pad_right), mode='reflect')
            lo_all = F.conv1d(fe_p, self.lo_filters)
            hi_all = F.conv1d(fo_p, self.hi_filters)

            # 融合
            w  = F.softmax(self.filter_weights, dim=0).view(1,1,-1,1)
            lo = (lo_all.view(B,C,self.num_filters,-1) * w).sum(2)
            hi = (hi_all.view(B,C,self.num_filters,-1) * w).sum(2)

            # lifting refine
            c, d = self.lifting_predictors[lvl](lo, hi)
            details[lvl] = d  # 更新
            current = c

            # 能量特征
            e_lo = lo.abs().mean(dim=[1,2], keepdim=True)  # (B,1,1)
            e_hi = hi.abs().mean(dim=[1,2], keepdim=True)
            energy_feats.append(torch.cat([e_lo, e_hi], dim=2).squeeze(1))  # (B,2)

        approx = current  # (B,1,L_final)
        energy_feats = torch.cat(energy_feats, dim=1)  # (B, levels*2)

        # 正则 + 约束
        regu = sum(d.abs().mean() * 0.1 for d in details)
        regu += 0.05 * torch.dist(approx.mean(), x.mean(), p=2)
        regu += self.wavelet_constraint_loss()

        # 归一化
        approx = self.norm_x(approx)
        details = [self.norm_d(d) for d in details]

        return approx, details, regu, energy_feats, trans_feats
    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        yl, yh, regu, energy_feats, trans_feats = self.forward(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]

        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l

class RevIN(nn.Module):
    """可逆实例归一化"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        mu  = x.mean(dim=[1,2], keepdim=True)
        sig = x.var(dim=[1,2], keepdim=True, unbiased=False).add(self.eps).sqrt()
        if mode=='norm':
            self.mu, self.sig = mu, sig
            return (x - mu)/sig * self.weight + self.bias
        else:
            return (x - self.bias)/(self.weight + self.eps) * self.sig + self.mu




import torch
import torch.nn as nn
import torch.nn.functional as F

class ReConstruction(nn.Module):
    def __init__(self, config, decomp):
        super().__init__()
        self.decomp = decomp
        feat_dim = config.level * 3
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.level * 2),
            nn.Sigmoid()
        )
        self.channel = config.c_in
        self.pred_length = config.pred_len
        self.recon_loss_w = getattr(config, 'recon_loss_w', 0.3)
        self.forecaster = nn.Sequential(
            nn.Conv1d(self.channel, config.d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.d_model, config.d_model // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.d_model // 2, self.channel, 3, padding=1)
        )
        self.detail_forecasters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.channel, config.d_model, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(config.d_model, self.channel, 3, padding=1)
            ) for _ in range(decomp.levels)
        ])
        self.revin = RevIN(self.channel)
    def frequency_interpolation(self,x,seq_len,target_len):
        len_ratio = seq_len/target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0),x_fft.size(1),target_len//2+1],dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:,:,:seq_len//2+1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2,n=target_len)
        out = out * len_ratio
        return out
    def forward(self, approx, details, regu, e_feats, t_feats, x_orig=None):
        B, C, _ = approx.shape
        feats = torch.cat([e_feats, t_feats], dim=1)
        gates = self.gate(feats).view(B, 1, self.decomp.levels, 2)
        for i in range(self.decomp.levels):
            details[i] = details[i] * gates[:, :, i, 1].unsqueeze(-1)
        approx = approx * gates[:, :, -1, 0].unsqueeze(-1)

        pred_lo = self.forecaster(approx)
        pred_lo = self.frequency_interpolation(pred_lo.to(torch.float32),target_len=self.pred_length,seq_len=pred_lo.size(-1))
        # pred_lo = F.interpolate(pred_lo, size=self.pred_length, mode='linear', align_corners=False)
        # pred_lo_den = self.revin(pred_lo, 'denorm') if x_orig is not None else pred_lo
        pred_lo_den = pred_lo

        pred_details = []
        for lvl, d in enumerate(details):
            d_pred = self.detail_forecasters[lvl](d)
            d_up = self.frequency_interpolation(d_pred.to(torch.float32), target_len=self.pred_length, seq_len=d_pred.size(-1))

            # d_up = F.interpolate(d_pred, size=self.pred_length, mode='linear', align_corners=False)
            # d_up = self.revin(d_up, 'denorm') if x_orig is not None else d_up
            pred_details.append(d_up)

        x_hat = self._inverse_reconstruct(pred_lo_den, pred_details)
        if x_orig is not None:
            loss_recon = F.mse_loss(x_hat, x_orig)
        else:


            loss_recon = F.mse_loss(pred_lo,
                                     self.frequency_interpolation(approx.to(torch.float32), target_len=self.pred_length, seq_len=approx.size(-1)))
        total_regu = regu + self.recon_loss_w * loss_recon
        return (x_hat, total_regu) if x_orig is not None else (pred_lo_den, total_regu)

    def _inverse_reconstruct(self, pred_lo, pred_details):
        current = pred_lo
        B, C, L = current.shape
        for lvl in reversed(range(self.decomp.levels)):
            # 确保 high-frequency 与 current 长度匹配
            d = pred_details[lvl]
            if d.shape[-1] != L:
                d = F.interpolate(d, size=L, mode='linear', align_corners=False)

            # reshape
            fe = current.reshape(B * C, 1, L)
            fo = d.reshape(B * C, 1, L)

            # padding
            ksz = self.decomp.ksz
            pad_left = ksz // 2 - 1
            pad_right = ksz // 2
            fe_p = F.pad(fe, (pad_left, pad_right), mode='reflect')
            fo_p = F.pad(fo, (pad_left, pad_right), mode='reflect')

            # inverse filters
            lo_all = F.conv1d(fe_p, self.decomp.rec_lo_filters)
            hi_all = F.conv1d(fo_p, self.decomp.rec_hi_filters)
            w = F.softmax(self.decomp.rec_filter_weights, dim=0).view(1, 1, -1, 1)
            lo = (lo_all.view(B, C, -1, lo_all.size(-1)) * w).sum(2)
            hi = (hi_all.view(B, C, -1, hi_all.size(-1)) * w).sum(2)

            # interleave
            out = torch.zeros(B, C, L * 2, device=current.device)
            out[:, :, ::2] = lo
            out[:, :, 1::2] = hi
            current = out
            L = current.size(-1)
        return current

class AdaptiveWaveletNet(nn.Module):
    """完整时间序列预测网络"""
    def __init__(self, config):
        super().__init__()
        self.decomp = LiftBoostLearnWave(
            init_wavelet=config.wavelet_name,
            num_filters=config.num_filters,
            levels=config.level,
            input_length=config.input_length,
            channel=config.channel,
            batch_size=config.batch_size,
            pred_length=config.pred_length,
            k_size=4,
            device=config.device,
        )

        self.construct = ReConstruction(config,self.decomp)
        # feat_dim = config['level'] * 3  # 2 energy + 1 transient per level
        # self.gate = nn.Sequential(
        #     nn.Linear(feat_dim, config['d_model']),
        #     nn.ReLU(),
        #     nn.Linear(config['d_model'], config['level']*2),
        #     nn.Sigmoid()
        # )
        # self.forecaster = nn.Sequential(
        #     nn.Conv1d(config['channel'], config['d_model'], 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(config['d_model'], config['d_model']//2, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(config['d_model']//2, config['channel'], 3, padding=1)
        # )
        # self.pred_length    = config['pred_length']
        # self.recon_loss_w   = config['recon_loss_weight']
        # self.revin          = RevIN(1)

    def forward(self, x):
        # 1. 归一化
        # x_norm = self.revin(x, 'norm')
        # 2. 分解
        approx, details, regu, e_feats, t_feats = self.decomp(x)

        # res = self.decomp.reconstruct(approx,details)
        res = self.construct(approx, details, regu, e_feats, t_feats)

        # plot_tensors([x] + [approx] + details,'asdasdasd')
        # plot_tensors([x] + [res] ,'asdasdasd1111111111')


        # 3. 门控
        # feats = torch.cat([e_feats, t_feats], dim=1)           # (B, level*3)
        # gates = self.gate(feats).view(-1,1,self.decomp.levels,2)
        # for i in range(self.decomp.levels):
        #     details[i] = details[i] * gates[:,:,i,1].unsqueeze(-1)
        # approx = approx * gates[:,:,-1,0].unsqueeze(-1)
        # # 4. 预测 + 上采样
        # pred = self.forecaster(approx)
        # pred_ups = F.interpolate(pred, size=self.pred_length,
        #                          mode='linear', align_corners=False)
        # pred_den = self.revin(pred_ups, 'denorm')
        # # 5. 重构损失（示例）
        # approx_up = F.interpolate(approx, size=self.pred_length,
        #                           mode='linear', align_corners=False)
        # recon_loss = F.mse_loss(pred_ups, approx_up)
        # return pred_den, regu + self.recon_loss_w * recon_loss
        return res
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
if __name__ == "__main__":
    # 检查 cuda 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 配置示例
    config = Config(
        wavelet_name='db4',
        num_filters=8,
        level=3,
        input_length=96,
        pred_length=192,
        batch_size=32,
        d_model=64,
        recon_loss_weight=0.3,
        channel=7,
        device=device,
        lambda_orth=0.01,
        lambda_energy=0.01,
        regu_details=0.1,
        regu_approx=0.05,
        pred_len=96,
        c_in=7
    )


    # 创建模型并移动到 device
    model = AdaptiveWaveletNet(config).to(device)
    # model.train()

    # 构造测试输入：batch_size=4, 通道=1, 长度=input_length
    batch_size = 32
    x = torch.randn(batch_size, config.channel, config.input_length, device=device)

    # 前向
    pred, loss = model(x)

    # 打印结果
    print(f"Input shape:  {x.shape}")
    print(f"Pred shape:   {pred.shape}   (should be [batch_size, 1, pred_length])")
    print(f"Loss value:   {loss.item():.6f}")

    # 测试反向传播
    loss.backward()
    print("Backward pass successful!")

# —— 科学性与创新性点评 ——
# • 你将“经典小波分解”与“可学习滤波器”+“提升（Lifting）机制”+“瞬态与能量特征门控”融合在一起，
#   在理论上同时兼顾了多尺度“频带分离”和数据驱动能力，**具有较强创新性**。
# • 峭度（超额峭度）和绝对能量被用作动态门控特征，能够自适应地强调或抑制各尺度信息，
#   这对非平稳时序的短期预测很有意义。
# • 同时加入了小波正交与能量约束，有助于学习出的滤波器保持数学性质，**提升了科学合理性**。
# • 如果你进一步加入“真正的逆小波重构”作为重构损失，而非简单插值，会让模型在理论上更严谨。

# 这样，模型既保留了小波在时频局部化的优势，又具备端到端可微学习的灵活性，在时间序列预测中是一种**颇具创新且科学合理**的设计。
