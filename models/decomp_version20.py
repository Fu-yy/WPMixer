import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class Splitting(nn.Module):
    def __init__(self, channel_first=True):
        super(Splitting, self).__init__()
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            return (x[:, :, ::2], x[:, :, 1::2])
        else:
            return (x[:, ::2, :], x[:, 1::2, :])


class AdaptiveLiftingStep(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters=1, simple_lifting=True):
        super(AdaptiveLiftingStep, self).__init__()
        self.num_filters = num_filters
        self.pad = (kernel_size // 2, kernel_size - 1 - kernel_size // 2)

        # 预测(P)和更新(U)网络
        self.P = self._build_module(in_channels, kernel_size, num_filters, simple_lifting)
        self.U = self._build_module(in_channels, kernel_size, num_filters, simple_lifting)

    def _build_module(self, in_channels, kernel_size, num_filters, simple_lifting):
        modules = [nn.ReflectionPad1d(self.pad)]
        if simple_lifting:
            modules.append(nn.Conv1d(in_channels, in_channels * num_filters,
                                     kernel_size=kernel_size, stride=1,
                                     groups=in_channels))
        else:
            hidden_size = 4
            modules.extend([
                nn.Conv1d(in_channels, in_channels * hidden_size,
                          kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.Conv1d(in_channels * hidden_size, in_channels * num_filters,
                          kernel_size=1, groups=in_channels)
            ])
        return nn.Sequential(*modules)

    def forward(self, x_even, x_odd, gate_weights=None, modified=True):
        # 预测和更新步骤
        P_out = self.P(x_even).view(x_even.size(0), x_even.size(1), self.num_filters, -1)
        U_out = self.U(x_odd).view(x_odd.size(0), x_odd.size(1), self.num_filters, -1)

        # 应用门控权重
        if gate_weights is not None:
            P_out = P_out * gate_weights.unsqueeze(1).unsqueeze(-1)
            U_out = U_out * gate_weights.unsqueeze(1).unsqueeze(-1)

        # 融合多滤波器结果
        P_fused = P_out.sum(dim=2)
        U_fused = U_out.sum(dim=2)

        # 提升方案计算
        if modified:
            c = x_even + U_fused
            d = x_odd - P_fused
        else:
            d = x_odd - P_fused
            c = x_even + U_fused

        return c, d


class AdaptiveWaveletDecomposition(nn.Module):
    def __init__(self, init_wavelet='db4', levels=3, num_filters=8,
                 kernel_size=4, d_model=64, regu_details=0.1, regu_approx=0.1,
                 lambda_orth=0.01, lambda_energy=0.01, input_length=None,
                 in_channels=1, device='cpu'):
        super().__init__()
        self.levels = levels
        self.num_filters = num_filters
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.lambda_orth = lambda_orth
        self.lambda_energy = lambda_energy
        self.device = device

        # 初始化提升步骤
        self.lifting_steps = nn.ModuleList([
            AdaptiveLiftingStep(in_channels, kernel_size, num_filters)
            for _ in range(levels)
        ]).to(self.device)

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(levels * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, levels * num_filters),
            nn.Softmax(dim=-1)
        ).to(self.device)

        # 信号分割
        self.split = Splitting(channel_first=True)
        # 为每一层创建独立的门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, d_model),  # 每层只有2个能量特征
                nn.ReLU(),
                nn.Linear(d_model, num_filters),
                nn.Softmax(dim=-1)
            ) for _ in range(levels)
        ]).to(self.device)

        # 存储门控权重用于重构
        # 存储门控值用于重构
        self.gates_reconstruct = [None] * levels

        # 预计算输出尺寸
        if input_length is not None:
            self.output_dims = self._dummy_forward(input_length, in_channels, device)

    def _dummy_forward(self, length, channels, device):
        x = torch.ones(1, channels, length, device=device)
        _, coeffs, _ = self.forward(x)
        return [c.shape[-1] for c in coeffs]

    def wavelet_constraint_loss(self):
        """正交性和能量约束损失"""
        orth_loss = 0
        energy_loss = 0

        for step in self.lifting_steps:
            # 提取卷积权重
            P_conv = step.P[1].weight if isinstance(step.P[0], nn.ReflectionPad1d) else step.P[0].weight
            U_conv = step.U[1].weight if isinstance(step.U[0], nn.ReflectionPad1d) else step.U[0].weight

            # 正交性约束
            for i in range(self.num_filters):
                p_kernel = P_conv[i::self.num_filters].view(self.num_filters, -1)
                u_kernel = U_conv[i::self.num_filters].view(self.num_filters, -1)
                orth_loss += torch.abs(torch.dot(p_kernel.flatten(), u_kernel.flatten()))

            # 能量约束
            energy_loss += torch.abs(torch.norm(P_conv) - 1) + torch.abs(torch.norm(U_conv) - 1)

        return self.lambda_orth * orth_loss + self.lambda_energy * energy_loss

    def forward(self, x):
        B, C, L = x.shape
        current = x
        coeffs = []
        energy_feats = []
        regu_loss = 0
        total_details = 0

        for i in range(self.levels):
            # 信号分割
            x_even, x_odd = self.split(current)

            # 计算当前层能量特征
            e_even = x_even.abs().mean(dim=(1, 2))  # [B]
            e_odd = x_odd.abs().mean(dim=(1, 2))  # [B]
            energy = torch.stack([e_even, e_odd], dim=1)  # [B, 2]
            # 获取门控权重
            # 获取当前层门控权重
            gate_weights = self.gates[i](energy)  # [B, num_filters]
            self.gates_reconstruct[i] = gate_weights.detach()

            # 自适应提升步骤
            # 自适应提升步骤
            c, d = self.lifting_steps[i](
                x_even, x_odd,
                gate_weights=gate_weights
            )

            # 正则化损失
            if self.regu_details > 0:
                regu_loss += self.regu_details * d.abs().mean()
                total_details += d.abs().sum()

            if self.regu_approx > 0:
                regu_loss += self.regu_approx * torch.dist(c.mean(), current.mean(), p=2)

            coeffs.append(d)
            current = c

        coeffs.append(current)
        # constraint_loss = self.wavelet_constraint_loss()
        constraint_loss = 0

        # 最终近似系数
        approx = coeffs[-1]

        return approx, coeffs[:-1], regu_loss + constraint_loss


class AdaptiveWaveletReconstruction(nn.Module):
    def __init__(self, decompose_module):
        super().__init__()
        self.levels = decompose_module.levels
        self.lifting_steps = decompose_module.lifting_steps
        self.gates_reconstruct = decompose_module.gates_reconstruct
        self.split = decompose_module.split

    def forward(self, approx, details):
        current = approx

        for i in range(self.levels - 1, -1, -1):
            d = details[i]
            gate_weights = self.gates_reconstruct[i]

            # 获取提升步骤模块
            lifting_step = self.lifting_steps[i]

            # 逆提升步骤（需要门控权重）
            if lifting_step.modified:
                # 计算P(current) - 多滤波器
                P_out = lifting_step.P(current)
                P_out = P_out.view(current.size(0), current.size(1),
                                   lifting_step.num_filters, -1)
                # 应用门控权重
                P_out = P_out * gate_weights.unsqueeze(1).unsqueeze(-1)
                P_fused = P_out.sum(dim=2)

                # 计算U(d) - 多滤波器
                U_out = lifting_step.U(d)
                U_out = U_out.view(d.size(0), d.size(1),
                                   lifting_step.num_filters, -1)
                U_out = U_out * gate_weights.unsqueeze(1).unsqueeze(-1)
                U_fused = U_out.sum(dim=2)

                x_odd = d + P_fused
                x_even = current - U_fused
            else:
                # 类似处理非修改模式...
                pass

            # 信号合并
            reconstructed = torch.zeros(x_even.size(0), x_even.size(1),
                                        x_even.size(2) * 2, device=x_even.device)
            reconstructed[:, :, ::2] = x_even
            reconstructed[:, :, 1::2] = x_odd
            current = reconstructed

        return current


class EnhancedWaveletBlock(nn.Module):
    """完整的小波处理块：分解+处理+重构"""

    def __init__(self, config):
        super().__init__()
        self.decompose = AdaptiveWaveletDecomposition(
            init_wavelet=config.init_wavelet,
            levels=config.levels,
            num_filters=config.num_filters,
            kernel_size=config.kernel_size,
            d_model=config.d_model,
            regu_details=config.regu_details,
            regu_approx=config.regu_approx,
            lambda_orth=config.lambda_orth,
            lambda_energy=config.lambda_energy,
            input_length=config.input_length,
            in_channels=config.enc_in,
            device=config.device
        )

        # 中间处理模块（示例：简单卷积）
        self.process_approx = nn.Sequential(
            nn.Conv1d(config.enc_in, config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model, config.enc_in, kernel_size=3, padding=1)
        )

        self.process_details = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config.enc_in, config.d_model // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(config.d_model // 2, config.enc_in, kernel_size=3, padding=1)
            ) for _ in range(config.levels)
        ])

        self.reconstruct = AdaptiveWaveletReconstruction(self.decompose)

    def forward(self, x):
        # 分解
        approx, details, regu_loss = self.decompose(x)

        # 处理系数
        approx = self.process_approx(approx)
        processed_details = []
        for i, det in enumerate(details):
            processed_details.append(self.process_details[i](det))

        # 重构
        reconstructed = self.reconstruct(approx, processed_details)

        return reconstructed, regu_loss


import torch
from torch import nn


# 定义配置类
class Config:
    def __init__(self):
        self.init_wavelet = 'db4'
        self.levels = 3
        self.num_filters = 8
        self.kernel_size = 4
        self.d_model = 64
        self.regu_details = 0.1
        self.regu_approx = 0.1
        self.lambda_orth = 0.01
        self.lambda_energy = 0.01
        self.input_length = 96
        self.enc_in = 7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 主函数
if __name__ == "__main__":
    # 创建配置对象
    config = Config()

    # 创建测试输入: (batch_size, seq_len, channels) -> (32, 96, 7)
    x = torch.randn(32, 96, 7)
    print(f"输入形状: {x.shape}")

    # 将输入转换为模型需要的格式: (batch, channels, seq_len)
    x = x.permute(0, 2, 1)  # 变为 (32, 7, 96)

    # 创建小波块
    wavelet_block = EnhancedWaveletBlock(config).to(config.device)
    x = x.to(config.device)

    # 前向传播
    reconstructed, regu_loss = wavelet_block(x)

    # 将输出转换回原始格式
    reconstructed = reconstructed.permute(0, 2, 1)  # 变为 (32, 96, 7)

    print(f"输出形状: {reconstructed.shape}")
    print(f"正则化损失: {regu_loss.item():.4f}")