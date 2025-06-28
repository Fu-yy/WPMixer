import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from einops import rearrange, reduce, repeat


# 改进1: 可学习小波适配器 (完全重写)
class LearnableWavelet(nn.Module):
    def __init__(self, init_wavelet='db4', num_filters=8, levels=3):
        super().__init__()
        self.levels = levels
        self.num_filters = num_filters

        # 初始化小波滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo).float()
        dec_hi = torch.tensor(wavelet.dec_hi).float()

        # 可学习滤波器参数
        self.lo_filters = nn.Parameter(torch.stack([dec_lo] * num_filters))
        self.hi_filters = nn.Parameter(torch.stack([dec_hi] * num_filters))
        self.filter_weights = nn.Parameter(torch.ones(num_filters))

        # 自适应门控 - 修正维度匹配问题
        self.gate = nn.Sequential(
            nn.Linear(levels * 2, 32),  # 输入维度: levels*2
            nn.ReLU(),
            nn.Linear(32, levels * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """多级可学习小波分解 - 最终修正版"""
        B, C, L = x.shape
        coeffs = []
        energy_features = []

        current = x
        for level_idx in range(self.levels):
            # 计算当前长度
            current_L = current.shape[2]

            # 多滤波器卷积
            lo_conv = torch.zeros(B, C, self.num_filters, current_L, device=x.device)
            hi_conv = torch.zeros(B, C, self.num_filters, current_L, device=x.device)

            # 对每个通道单独处理
            for c_idx in range(C):
                channel_data = current[:, c_idx:c_idx + 1, :]

                # 低通滤波
                lo_conv_c = F.conv1d(
                    channel_data,
                    self.lo_filters.unsqueeze(1),
                    padding=(self.lo_filters.shape[1] // 2)
                )

                # 高通滤波
                hi_conv_c = F.conv1d(
                    channel_data,
                    self.hi_filters.unsqueeze(1),
                    padding=(self.hi_filters.shape[1] // 2)
                )

                # 确保长度一致
                lo_conv_c = F.pad(lo_conv_c, (0, current_L - lo_conv_c.shape[2]))
                hi_conv_c = F.pad(hi_conv_c, (0, current_L - hi_conv_c.shape[2]))

                lo_conv[:, c_idx] = lo_conv_c.squeeze(1)
                hi_conv[:, c_idx] = hi_conv_c.squeeze(1)

            # 滤波器加权融合
            weights = F.softmax(self.filter_weights, dim=0)
            lo_fused = torch.sum(lo_conv * weights.view(1, 1, -1, 1), dim=2)
            hi_fused = torch.sum(hi_conv * weights.view(1, 1, -1, 1), dim=2)

            # 下采样 (取偶数索引)
            approx = lo_fused[:, :, ::2]
            detail = hi_fused[:, :, ::2]

            coeffs.append(detail)
            current = approx

            # 能量特征提取 - 修正维度问题
            # 每个级别提取每个样本的全局能量特征
            energy_approx = approx.abs().mean(dim=[1, 2], keepdim=True)  # (B, 1)
            energy_detail = detail.abs().mean(dim=[1, 2], keepdim=True)  # (B, 1)
            energy = torch.cat([energy_approx, energy_detail], dim=-1)  # (B, 2)
            energy_features.append(energy)

        coeffs.append(current)  # 最后一级的近似系数

        # 合并所有级别的能量特征
        energy_features = torch.cat(energy_features, dim=-1)  # (B, levels*2)

        # 频段重要性门控
        gate_weights = self.gate(energy_features)  # (B, levels*2)
        gate_weights = gate_weights.view(B, 1, self.levels, 2)  # (B, 1, levels, 2)

        # 应用门控权重
        for i in range(self.levels):
            coeffs[i] = coeffs[i] * gate_weights[:, :, i, 1].unsqueeze(-1)
        coeffs[-1] = coeffs[-1] * gate_weights[:, :, -1, 0].unsqueeze(-1)

        return coeffs


class LearnableWaveletNew(nn.Module):
    def __init__(self, init_wavelet='db4', num_filters=8, levels=3,input_length=None,pred_length=None, batch_size=1, channel=1,d_model=None, device='cpu'):
        super().__init__()
        self.levels = levels
        self.num_filters = num_filters

        # 初始化小波分解低/高通滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo).float()  # 低通分解
        dec_hi = torch.tensor(wavelet.dec_hi).float()  # 高通分解
        self.d_model=d_model
        # 可学习滤波器组 [F, K]
        self.lo_filters = nn.Parameter(torch.stack([dec_lo] * num_filters))
        self.hi_filters = nn.Parameter(torch.stack([dec_hi] * num_filters))
        self.filter_weights = nn.Parameter(torch.ones(num_filters))
        self.device = device

        # 能量特征门控网络：输入 levels*2，输出 levels*2
        self.gate = nn.Sequential(
            nn.Linear(levels * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, levels * 2),
            nn.Sigmoid()
        )

        # 预计算一维卷积左右填充量
        self.lo_pad = dec_lo.shape[0] // 2
        self.hi_pad = dec_hi.shape[0] // 2

        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length,batch_size,channel,self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length,batch_size,channel,self.device)  # required length of the pred seq after decom

    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        yl, yh = self.forward(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]

        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l

    def forward(self, x):
        """
        Args:
            x: [B, C, L] 输入信号
        Returns:
            coeffs: list of length levels+1
                - coeffs[0..levels-1]: 各级 detail 系数 [B, C, L//2]
                - coeffs[-1]: 最后一级 approximation 系数 [B, C, L//2]
        """
        B, C, L = x.shape
        coeffs = []
        energy_feats = []

        # 将滤波器 reshape 成 conv2d 所需格式：
        # in_channels=1, out_channels=num_filters
        # [F, K] -> [F, 1, 1, K]
        lo_kernel = self.lo_filters.view(self.num_filters, 1, 1, -1)
        hi_kernel = self.hi_filters.view(self.num_filters, 1, 1, -1)

        current = x
        for lvl in range(self.levels):
            cur_L = current.size(2)
            # [B, C, L] -> [B, 1, C, L]
            inp = current.view(B, 1, C, cur_L)

            # 并行对所有通道做 num_filters 个卷积
            lo_out = F.conv2d(inp, lo_kernel, padding=(0, self.lo_pad))
            hi_out = F.conv2d(inp, hi_kernel, padding=(0, self.hi_pad))
            # shapes: [B, F, C, L]

            # 转回 [B, C, F, L]
            lo_out = lo_out.permute(0, 2, 1, 3)
            hi_out = hi_out.permute(0, 2, 1, 3)

            # 如果长度不足，右侧补零
            if lo_out.size(3) < cur_L:
                diff = cur_L - lo_out.size(3)
                lo_out = F.pad(lo_out, (0, diff))
                hi_out = F.pad(hi_out, (0, diff))

            # 按 filter_weights 加权融合
            w = F.softmax(self.filter_weights, dim=0)  # [F]
            lo_fused = torch.einsum('bcfl,f->bcl', lo_out, w)  # [B, C, L]
            hi_fused = torch.einsum('bcfl,f->bcl', hi_out, w)  # [B, C, L]

            # 下采样得到 approximation 和 detail
            approx = lo_fused[:, :, ::2]   # [B, C, L//2]
            detail = hi_fused[:, :, ::2]   # [B, C, L//2]

            coeffs.append(detail)
            current = approx

            # 记录能量特征 (approx, detail)
            e_a = approx.abs().mean(dim=(1,2), keepdim=True)  # [B,1,1]
            e_d = detail.abs().mean(dim=(1,2), keepdim=True)  # [B,1,1]
            energy_feats.append(torch.cat([e_a, e_d], dim=-1))  # [B,1,2]

        # 最后一级的 approximation
        coeffs.append(current)  # [B, C, L//(2^levels)]

        # 拼接所有能量特征并送入门控网络
        # energy_feats: list of levels tensors [B,1,2] -> [B, levels*2]
        energy = torch.cat(energy_feats, dim=-1).view(B, -1)
        gate_out = self.gate(energy)                   # [B, levels*2]
        gate_w = gate_out.view(B, 1, self.levels, 2)   # [B,1,levels,2]

        # 应用门控：detail 用 gate_w[...,i,1]，最后一级 approx 用 gate_w[..., -1,0]
        for i in range(self.levels):
            coeffs[i] = coeffs[i] * gate_w[:, :, i, 1].unsqueeze(-1)
        coeffs[-1] = coeffs[-1] * gate_w[:, :, -1, 0].unsqueeze(-1)

        return coeffs[-1], coeffs[:-1]

# 改进2: 多尺度时序卷积

class LearnableWavelet_new_new(nn.Module):
    """
    Learnable multi-level wavelet decomposition module.
    Initializes learnable low-pass and high-pass filters based on a given pywt wavelet,
    applies them in parallel across channels and custom number of filters,
    fuses via learned weights, and uses stride for downsampling.

    Args:
        init_wavelet (str): name of the initial wavelet (e.g., 'db4').
        num_filters (int): number of parallel learnable filters.
        levels (int): number of decomposition levels.
    """

    def __init__(self, init_wavelet='db4', num_filters=8, levels=3,input_length=None,pred_length=None, batch_size=1, channel=1,d_model=None, device='cpu',lambda_orth=0.01, lambda_energy=0.01):
        super().__init__()
        self.levels = levels
        self.num_filters = num_filters
        self.d_model = d_model
        # 初始化小波滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo).float()  # (kernel_size,)
        dec_hi = torch.tensor(wavelet.dec_hi).float()
        kernel_size = dec_lo.numel()
        self.device = device
        self.lambda_orth = lambda_orth
        self.lambda_energy = lambda_energy
        # 可学习滤波器参数: (num_filters, 1, kernel_size)
        self.lo_filters = nn.Parameter(
            dec_lo.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1), requires_grad=True
        )
        self.hi_filters = nn.Parameter(
            dec_hi.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1), requires_grad=True
        )
        self.filter_weights = nn.Parameter(torch.ones(num_filters), requires_grad=True)

        # 分析门控网络
        self.gate = nn.Sequential(
            nn.Linear(levels * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, levels * 2),
            nn.Sigmoid()
        )

        # 固定 padding 和 stride
        self.pad = kernel_size // 2
        self.pad_rec = kernel_size // 2   # 重构padding
        self.stride = 2
        self.output_padding = 0  # 重构输出填充

        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length, batch_size, channel,
                                               self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length, batch_size, channel,
                                              self.device)  # required length of the pred seq after decom
        # 存储门控值用于重构
        self.gates_reconstruct = None



    def wavelet_constraint_loss(self):
        # 正交性约束
        orth_loss = 0
        for i in range(self.num_filters):
            dot_product = torch.dot(self.lo_filters[i].squeeze(), self.hi_filters[i].squeeze())
            orth_loss += torch.abs(dot_product)

        # 单位能量约束
        energy_loss = 0
        for i in range(self.num_filters):
            lo_energy = torch.norm(self.lo_filters[i]) - 1
            hi_energy = torch.norm(self.hi_filters[i]) - 1
            energy_loss += torch.abs(lo_energy) + torch.abs(hi_energy)

        return self.lambda_orth * orth_loss + self.lambda_energy * energy_loss
    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        yl, yh,_ = self.forward(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]

        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, C, L)
        Returns:
            coeffs (list of torch.Tensor): list of detail coefficients for each level
                followed by the final approximation tensor.
        """
        B, C, L = x.shape
        coeffs = []
        energy_feats = []
        self.gates_reconstruct = None
        current = x
        # 多级分解
        for _ in range(self.levels):
            B, C, L = current.shape
            # 展平通道以批量并行 conv
            flat = torch.reshape(current, (B * C, 1, L))

            # 并行 low-pass 和 high-pass 卷积 + downsample
            lo_all = F.conv1d(flat, self.lo_filters, padding=self.pad, stride=self.stride)
            hi_all = F.conv1d(flat, self.hi_filters, padding=self.pad, stride=self.stride)

            # 恢复到 (B, C, num_filters, L_down)
            Ld = lo_all.size(-1)
            lo_all = lo_all.view(B, C, self.num_filters, Ld)
            hi_all = hi_all.view(B, C, self.num_filters, Ld)

            # 根据 filter_weights 融合
            w = F.softmax(self.filter_weights, dim=0).view(1, 1, self.num_filters, 1)
            lo = (lo_all * w).sum(dim=2)  # (B, C, L_down)
            hi = (hi_all * w).sum(dim=2)

            # 保存细节系数，低频进入下一层
            coeffs.append(hi)
            current = lo

            # 能量特征: (B,1,1) -> (B,2)
            e_lo = lo.abs().mean(dim=[1,2], keepdim=True)
            e_hi = hi.abs().mean(dim=[1,2], keepdim=True)
            energy_feats.append(torch.cat([e_lo, e_hi], dim=2).squeeze(1))

        # 最后一级近似系数
        coeffs.append(current)

        # 拼接能量特征 (B, levels*2)
        energy = torch.cat(energy_feats, dim=1)

        # 频段重要性门控
        gates = self.gate(energy)  # (B, levels*2)
        gates = gates.view(B, 1, self.levels, 2)
        self.gates_reconstruct = gates  # 存储门控值用于重构

        # 施加门控
        for i in range(self.levels):
            coeffs[i] = coeffs[i] * gates[:, :, i, 1].unsqueeze(-1)
        # 最后一项用 第i层的 approximation gate
        coeffs[-1] = coeffs[-1] * gates[:, :, -1, 0].unsqueeze(-1)
        loss = self.wavelet_constraint_loss()
        return coeffs[-1], coeffs[:-1],loss



class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks, d in zip(kernel_sizes, dilations):
            padding = (ks - 1) * d // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, ks,
                              padding=padding, dilation=d),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU()
                )
            )
        self.fusion = nn.Conv1d(len(self.convs) * out_channels, out_channels, 1)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)


# 改进3: 跨分辨率注意力
class CrossResolutionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branch_features):
        """branch_features: 各分支特征列表 [ (B, N_i, d) ] """
        # 特征对齐
        max_len = max([f.shape[1] for f in branch_features])
        aligned_features = []
        for f in branch_features:
            if f.shape[1] < max_len:
                # 插值对齐
                f = F.interpolate(f.permute(0, 2, 1), size=max_len,
                                  mode='linear').permute(0, 2, 1)
            aligned_features.append(f)

        # 拼接特征
        x = torch.stack(aligned_features, dim=1)  # (B, num_branches, max_len, d)
        B, NB, L, D = x.shape

        # 多头注意力
        q = self.q_proj(x).view(B, NB, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k_proj(x).view(B, NB, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(x).view(B, NB, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # 注意力计算
        attn = torch.einsum('bhnld,bhmld->bhnlm', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnlm,bhmld->bhnld', attn, v)
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, NB, L, D)
        out = self.out_proj(out)

        # 恢复原始尺寸
        processed_features = []
        for i, f in enumerate(branch_features):
            orig_len = f.shape[1]
            if orig_len < max_len:
                # 下采样回原始长度
                resized = F.interpolate(out[:, i].permute(0, 2, 1),
                                        size=orig_len, mode='linear').permute(0, 2, 1)
            else:
                resized = out[:, i]
            processed_features.append(resized)

        return processed_features


# 改进4: 轻量化混合器
class LiteMixerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        inner_dim = int(embed_dim * expansion_factor)

        # 共享层
        self.shared_norm = nn.LayerNorm(embed_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # 分支特定层
        self.branch_norm = nn.LayerNorm(embed_dim)
        self.branch_mlp = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """x: (B, N, d)"""
        # 跨分支共享信息
        shared = reduce(x, 'b n d -> b d', 'mean')
        shared = self.shared_mlp(self.shared_norm(shared))
        shared = repeat(shared, 'b d -> b n d', n=x.shape[1])

        # 分支特定处理
        branch_out = self.branch_mlp(self.branch_norm(x))

        return branch_out + shared


# 核心模块: 增强型WPMixer分支
class EnhancedBranch(nn.Module):
    def __init__(self, input_len, pred_len, embed_dim=64, patch_size=8, stride=4,
                 expansion_factor=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

        # 多尺度卷积
        self.multiscale_conv = MultiScaleConv(1, embed_dim // 2)

        # 可逆实例归一化
        self.revin = nn.LayerNorm(embed_dim)

        # 分块和嵌入
        num_patches = (input_len - patch_size) // stride + 1
        self.patch_embed = nn.Conv1d(1, embed_dim, patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # 轻量化混合器
        self.mixers = nn.Sequential(
            LiteMixerBlock(embed_dim, expansion_factor, dropout),
            LiteMixerBlock(embed_dim, expansion_factor, dropout)
        )

        # 输出头
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_patches * embed_dim, pred_len)
        )

    def forward(self, x):
        """x: (B, C, L)"""
        B, C, L = x.shape
        # 多尺度特征提取
        conv_feat = self.multiscale_conv(x)

        # 分块嵌入
        patches = self.patch_embed(x)  # (B, d, N)
        patches = patches.permute(0, 2, 1)  # (B, N, d)
        patches = patches + self.pos_embed

        # 与卷积特征融合
        conv_feat = conv_feat.permute(0, 2, 1)
        patches = torch.cat([patches, conv_feat], dim=-1)

        # 归一化
        patches = self.revin(patches)

        # 混合器处理
        mixed = self.mixers(patches)

        # 预测
        return self.head(mixed)


# 完整模型
class EnhancedWPMixer(nn.Module):
    def __init__(self, seq_len, pred_len, channels=1, wavelet_levels=3,
                 embed_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.wavelet_levels = wavelet_levels

        # 输入归一化
        self.input_revin = nn.LayerNorm(channels)

        # 可学习小波分解
        self.wavelet_decomp = LearnableWavelet(levels=wavelet_levels)

        # 分支预测器
        self.branches = nn.ModuleList()
        for i in range(wavelet_levels + 1):
            # 计算每个分支的输入长度
            input_len = seq_len // (2 ** (i + 1))
            branch = EnhancedBranch(input_len, pred_len, embed_dim)
            self.branches.append(branch)

        # 跨分辨率注意力
        self.cross_attn = CrossResolutionAttention(embed_dim)

        # 小波重构适配器
        self.recon_weights = nn.Parameter(torch.ones(wavelet_levels + 1))

        # 输出反归一化
        self.denorm = nn.LayerNorm(channels)

    def forward(self, x):
        """x: (B, L, C)"""
        B, L, C = x.shape
        # 输入归一化
        x = self.input_revin(x)
        x = x.permute(0, 2, 1)  # (B, C, L)

        # 小波分解
        coeffs = self.wavelet_decomp(x)

        # 各分支处理
        branch_outputs = []
        branch_features = []
        for i, coeff in enumerate(coeffs):
            # 确保系数有正确的形状 (B, C, L_i)
            if len(coeff.shape) == 2:
                coeff = coeff.unsqueeze(1)

            # 检查长度是否匹配分支预期
            expected_len = self.seq_len // (2 ** (i + 1))
            if coeff.shape[2] != expected_len:
                coeff = F.interpolate(coeff, size=expected_len, mode='linear')

            # 分支预测
            pred = self.branches[i](coeff)  # (B, C, T)
            branch_outputs.append(pred)

            # 提取特征用于跨分辨率交互
            patches = self.branches[i].patch_embed(coeff).permute(0, 2, 1)
            patches = self.branches[i].revin(patches)
            feat = self.branches[i].mixers[0](patches)
            branch_features.append(feat)

        # 跨分辨率交互
        enhanced_features = self.cross_attn(branch_features)

        # 特征增强的预测
        enhanced_outputs = []
        for i, feat in enumerate(enhanced_features):
            # 简单投影增强
            delta = self.branches[i].head(feat.flatten(1))
            enhanced_outputs.append(branch_outputs[i] + delta)

        # 加权重构
        weights = F.softmax(self.recon_weights, dim=0)
        recon = torch.zeros(B, C, self.pred_len, device=x.device)
        for i, out in enumerate(enhanced_outputs):
            # 确保输出形状正确
            if len(out.shape) == 2:
                out = out.unsqueeze(1)

            # 上采样到预测长度
            if out.shape[-1] != self.pred_len:
                out = F.interpolate(out, size=self.pred_len, mode='linear')
            recon += weights[i] * out

        # 反归一化
        recon = recon.permute(0, 2, 1)  # (B, T, C)
        return self.denorm(recon)


# 频域增强损失函数
class EnhancedLoss(nn.Module):
    def __init__(self, wavelet='db4', levels=3, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.levels = levels
        self.wavelet = wavelet

        # 注册小波滤波器作为缓冲区
        wavelet = pywt.Wavelet(wavelet)
        self.register_buffer('dec_lo', torch.tensor(wavelet.dec_lo).float())
        self.register_buffer('dec_hi', torch.tensor(wavelet.dec_hi).float())

    def wavelet_loss(self, pred, target):
        """计算小波系数损失"""
        loss = 0
        current_pred = pred
        current_target = target

        for i in range(self.levels):
            # 目标分解
            target_lo = F.conv1d(current_target, self.dec_lo.view(1, 1, -1), stride=2)
            target_hi = F.conv1d(current_target, self.dec_hi.view(1, 1, -1), stride=2)

            # 预测分解
            pred_lo = F.conv1d(current_pred, self.dec_lo.view(1, 1, -1), stride=2)
            pred_hi = F.conv1d(current_pred, self.dec_hi.view(1, 1, -1), stride=2)

            # 高频细节损失
            loss += F.l1_loss(pred_hi, target_hi)

            # 准备下一级分解
            current_target = target_lo
            current_pred = pred_lo

        return loss / self.levels

    def forward(self, pred, target):
        """组合时域和频域损失"""
        time_loss = F.smooth_l1_loss(pred, target)
        freq_loss = self.wavelet_loss(pred.permute(0, 2, 1), target.permute(0, 2, 1))
        return (1 - self.alpha) * time_loss + self.alpha * freq_loss


# 测试示例
if __name__ == "__main__":
    # 配置参数
    seq_len = 96
    pred_len = 24
    channels = 3
    batch_size = 16

    # 创建模型
    model = EnhancedWPMixer(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        wavelet_levels=3,
        embed_dim=64
    )

    # 损失函数
    criterion = EnhancedLoss(levels=3, alpha=0.3)

    # 示例输入
    x = torch.randn(batch_size, seq_len, channels)
    y = torch.randn(batch_size, pred_len, channels)

    # 前向传播
    pred = model(x)
    print(f"预测形状: {pred.shape}")  # 应为 [16, 24, 3]

    # 损失计算
    loss = criterion(pred, y)
    print(f"损失值: {loss.item():.4f}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params / 1e6:.2f}M")