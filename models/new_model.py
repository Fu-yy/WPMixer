import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class EnhancedGate(nn.Module):
    """
    增强门控机制：结合局部特征和全局注意力
    """

    def __init__(self, in_channels, d_model, levels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, levels * 2),
            nn.Sigmoid()
        )
        self.levels = levels

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, L)
        Returns:
            gates: 门控值 (B, levels*2)
        """
        # 提取局部特征
        local_feat = self.conv(x)  # (B, d_model, L)
        local_feat = local_feat.transpose(1, 2)  # (B, L, d_model)

        # 全局注意力
        global_feat, _ = self.attention(local_feat, local_feat, local_feat)
        global_feat = global_feat.mean(dim=1)  # (B, d_model)

        # 生成门控值
        gates = self.fc(global_feat)  # (B, levels*2)
        return gates


class LearnableWaveletEnhanced(nn.Module):
    """
    增强版可学习小波分解模块
    - 改进门控机制：结合局部特征和全局注意力
    - 添加滤波器约束：正交性和单位能量约束
    - 引入多尺度特征交互

    Args:
        init_wavelet (str): 初始化小波基名称 (如 'db4')
        num_filters (int): 并行滤波器数量
        levels (int): 分解层级
        input_length (int): 输入序列长度
        pred_length (int): 预测序列长度
        batch_size (int): 批大小
        channel (int): 输入通道数
        d_model (int): 模型维度
        device (str): 计算设备
        lambda_orth (float): 正交性约束权重
        lambda_energy (float): 能量约束权重
    """

    def __init__(self, init_wavelet='db4', num_filters=8, levels=3,
                 input_length=None, pred_length=None, batch_size=1,
                 channel=1, d_model=None, device='cpu',
                 lambda_orth=0.01, lambda_energy=0.01):
        super().__init__()
        self.levels = levels
        self.num_filters = num_filters
        self.d_model = d_model
        self.lambda_orth = lambda_orth
        self.lambda_energy = lambda_energy

        # 初始化小波滤波器
        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo).float()
        dec_hi = torch.tensor(wavelet.dec_hi).float()
        kernel_size = dec_lo.numel()
        self.device = device

        # 可学习滤波器参数
        self.lo_filters = nn.Parameter(
            dec_lo.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1)
        )
        self.hi_filters = nn.Parameter(
            dec_hi.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1)
        )
        self.filter_weights = nn.Parameter(torch.ones(num_filters))

        # 重构滤波器
        # rec_lo = dec_lo.flip(0)
        # rec_hi = dec_hi.flip(0)
        # self.lo_filters_rec = nn.Parameter(
        #     rec_lo.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1)
        # self.hi_filters_rec = nn.Parameter(
        #     rec_hi.unsqueeze(0).unsqueeze(1).repeat(num_filters, 1, 1)
        # self.filter_weights_rec = nn.Parameter(torch.ones(num_filters))

        # 增强门控网络
        self.gate = EnhancedGate(channel, d_model, levels)

        # 多尺度特征交互模块
        self.cross_scale = nn.ModuleList()
        for i in range(1, levels):
            self.cross_scale.append(
                nn.Conv1d(channel, channel, kernel_size=3, padding=1)
            )

        # 固定padding和stride
        self.pad = kernel_size // 2 - 1
        self.pad_rec = kernel_size // 2 - 1
        self.stride = 2
        self.output_padding = 0

        # 预计算输出尺寸
        self.input_w_dim = self._dummy_forward(input_length, batch_size, channel, device)
        self.pred_w_dim = self._dummy_forward(pred_length, batch_size, channel, device)
        self.gates_reconstruct = None

    def _dummy_forward(self, length, batch_size, channel, device):
        """预计算输出维度"""
        x = torch.ones(batch_size, channel, length, device=device)
        yl, yh = self.forward(x)
        l = [yl.shape[-1]]
        for c in yh:
            l.append(c.shape[-1])
        return l

    def wavelet_constraint_loss(self):
        """计算小波约束损失: 正交性 + 单位能量"""
        # 正交性约束: 低通和高通滤波器应正交
        orth_loss = 0
        for i in range(self.num_filters):
            lo = self.lo_filters[i].squeeze()
            hi = self.hi_filters[i].squeeze()
            dot_product = torch.dot(lo, hi)
            orth_loss += torch.abs(dot_product)

        # 单位能量约束: 滤波器能量应为1
        energy_loss = 0
        for i in range(self.num_filters):
            lo_energy = torch.norm(self.lo_filters[i]) - 1
            hi_energy = torch.norm(self.hi_filters[i]) - 1
            energy_loss += torch.abs(lo_energy) + torch.abs(hi_energy)

        return self.lambda_orth * orth_loss + self.lambda_energy * energy_loss

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 输入张量 (B, C, L)
        Returns:
            coeffs: 分解系数列表 [approx, detail1, detail2, ...]
        """
        B, C, L = x.shape
        coeffs = []
        prev_feat = None  # 存储上一级特征
        current = x

        # 多级分解
        for level in range(self.levels):
            B, C, L = current.shape

            # 多尺度特征交互
            # if level > 0 and prev_feat is not None:
            #     # 调整上一级特征尺寸
            #     prev_feat_resized = F.interpolate(prev_feat, size=L, mode='linear')
            #     # 特征融合
            #     current = current + self.cross_scale[level - 1](prev_feat_resized)

            # 展平通道进行批量卷积
            flat = torch.reshape(current, (B * C, 1, L))

            # 并行卷积
            lo_all = F.conv1d(flat, self.lo_filters, padding=self.pad, stride=self.stride)
            hi_all = F.conv1d(flat, self.hi_filters, padding=self.pad, stride=self.stride)

            # 恢复形状
            Ld = lo_all.size(-1)
            lo_all = lo_all.view(B, C, self.num_filters, Ld)
            hi_all = hi_all.view(B, C, self.num_filters, Ld)

            # 融合滤波器结果
            w = F.softmax(self.filter_weights, dim=0).view(1, 1, self.num_filters, 1)
            lo = (lo_all * w).sum(dim=2)  # (B, C, L_down)
            hi = (hi_all * w).sum(dim=2)

            # 保存细节系数
            coeffs.append(hi)
            prev_feat = current  # 保存当前级特征
            current = lo  # 低频进入下一层

        # 最后一级近似系数
        coeffs.append(current)

        # 增强门控
        gates = self.gate(x)  # (B, levels*2)
        gates = gates.view(B, 1, self.levels, 2)
        self.gates_reconstruct = gates

        # 应用门控
        for i in range(self.levels):
            coeffs[i] = coeffs[i] * gates[:, :, i, 1].unsqueeze(-1)
        coeffs[-1] = coeffs[-1] * gates[:, :, -1, 0].unsqueeze(-1)

        return coeffs[-1], coeffs[:-1]  # (approx, details)

    def reconstruct(self, approx, details):
        """
        小波重构
        Args:
            approx: 近似系数 (B, C, L)
            details: 细节系数列表 [detail1, detail2, ...]
        Returns:
            重构信号 (B, C, L)
        """
        current = approx
        details = details[::-1]  # 从最低频到最高频

        for i in range(self.levels):
            # 应用门控逆操作
            gate_val = self.gates_reconstruct[:, :, -(i + 1), 0].unsqueeze(-1)
            current = current / (gate_val + 1e-8)

            # 上采样
            L_up = details[i].shape[-1] * 2
            current_up = F.interpolate(current, size=L_up, mode='linear')

            # 卷积重构
            flat = torch.reshape(current_up, (B * C, 1, L_up))

            # 并行卷积
            lo_rec_all = F.conv1d(flat, self.lo_filters_rec, padding=self.pad_rec)
            hi_rec_all = F.conv1d(flat, self.hi_filters_rec, padding=self.pad_rec)

            # 恢复形状
            lo_rec_all = lo_rec_all.view(B, C, self.num_filters, L_up)
            hi_rec_all = hi_rec_all.view(B, C, self.num_filters, L_up)

            # 融合滤波器结果
            w = F.softmax(self.filter_weights_rec, dim=0).view(1, 1, self.num_filters, 1)
            lo_rec = (lo_rec_all * w).sum(dim=2)
            hi_rec = (hi_rec_all * w).sum(dim=2)

            # 重构信号 = 低频 + 高频
            current = lo_rec + hi_rec

            # 添加细节系数（应用门控逆操作）
            detail_gate_val = self.gates_reconstruct[:, :, -(i + 1), 1].unsqueeze(-1)
            current = current + details[i] / (detail_gate_val + 1e-8)

        return current