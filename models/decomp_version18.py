import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np


# ====================== 创新点1: 增强型动态小波基 (DyWAN-Pro) ======================
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

        # 小波生成器（带频率约束）
        self.wavelet_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, filter_length * 2)  # 低通+高通
        )

        # 频带约束初始化
        self.register_buffer('freq_constraint',
                             torch.linspace(0, 1, filter_length).view(1, -1))

    def forward(self, x):
        # x: [batch, channel, seq_len]
        stat_feat = self.stat_net(x)  # [batch, hidden]
        filters = self.wavelet_generator(stat_feat)  # [batch, 2*filter_length]
        lo_f = filters[:, :self.filter_length]  # [batch, filter_length]
        hi_f = filters[:, self.filter_length:]  # [batch, filter_length]

        # 频带约束：低通滤波器应平滑
        kernel = torch.tensor([[-1, 1]], dtype=torch.float32).view(1, 1, 2).to(lo_f.device)
        lo_smooth = F.conv1d(lo_f.unsqueeze(1), kernel, padding=1).abs().mean()

        # 正交性约束（简化版）
        ortho_loss = self._orthogonality_constraint(lo_f, hi_f)

        return lo_f, hi_f, ortho_loss + lo_smooth * 0.1

    def _orthogonality_constraint(self, lo_f, hi_f):
        # 增加正则化强度
        lo_norm = lo_f / (torch.norm(lo_f, dim=1, keepdim=True) + 1e-8)
        hi_norm = hi_f / (torch.norm(hi_f, dim=1, keepdim=True) + 1e-8)

        # 增加移位正交约束的阶数
        identity = torch.eye(self.filter_length, device=lo_f.device)
        shift_loss = 0
        for shift in range(1, 4):  # 检查更多移位情况
            shifted = torch.roll(lo_norm, shifts=shift, dims=1)
            shift_loss += torch.abs(
                torch.bmm(lo_norm.unsqueeze(2), shifted.unsqueeze(1))
            ).mean()

        # 增加幅值约束
        amp_loss = torch.abs(lo_norm.pow(2).sum(dim=1) - 1).mean()

        return self.reg_factor * (shift_loss + amp_loss)


# ====================== 物理感知激活函数 ======================
class PhysicsActivation(nn.Module):
    """物理感知的激活函数"""

    def forward(self, x):
        # 小波域应保持的数学特性
        return x * torch.sigmoid(x)  # 自适应门控

    def energy_constraint(self, input, output):
        """能量守恒约束"""
        energy_in = torch.mean(input ** 2, dim=(1, 2))
        energy_out = torch.mean(output ** 2, dim=(1, 2))
        return torch.abs(energy_in - energy_out).mean()


# ====================== 尺度感知ODE ======================
class ScaleAwareODE(nn.Module):
    """尺度感知的神经微分方程"""

    def __init__(self, dim, hidden, time_dim):
        super().__init__()
        # 确保隐藏层维度能被输入维度整除
        if hidden % dim != 0:
            hidden = dim * (hidden // dim + 1)

        # 频率特性相关的网络结构
        self.freq_net = nn.Sequential(
            nn.Conv1d(dim, hidden, 3, padding=1, groups=1),  # 修改为 groups=1
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=1)  # 修改为 groups=1
        )

        # 时间调制网络
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU()
        )
        self.time_mod = nn.Linear(time_dim, hidden * 2)

        # 尺度条件参数
        self.scale_embed = nn.Embedding(8, hidden)  # 假设最多8个尺度

        # 物理感知激活
        self.phys_act = PhysicsActivation()

    def forward(self, t, x, scale):
        # x形状: [B, C, L]
        B, C, L = x.shape

        # 将尺度索引转换为张量
        if not isinstance(scale, torch.Tensor):
            # 创建与批次大小匹配的尺度张量
            scale_tensor = torch.full((B,), scale, dtype=torch.long, device=x.device)
        else:
            scale_tensor = scale

        # 提取频率特征
        freq_feat = self.freq_net(x)

        # 时间调制
        t_emb = self.time_embed(t.view(1, 1)).expand(B, -1)
        gamma, beta = self.time_mod(t_emb).chunk(2, dim=1)
        gamma = gamma.view(B, -1, 1)
        beta = beta.view(B, -1, 1)

        # 尺度调制
        scale_emb = self.scale_embed(scale_tensor).view(B, -1, 1)

        # 调制特征
        modulated = (freq_feat * (1 + gamma + scale_emb)) + beta

        # 物理约束输出
        return self.phys_act(modulated) - 0.1 * x  # 带阻尼项


# ====================== 创新点3: 跨尺度融合机制 (CSF) ======================
class CrossScaleFusion(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.levels = levels

        # 门控机制控制信息流
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(levels)
        ])

        # 尺度间注意力
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels // 4, 1),
                nn.GELU(),
                nn.Conv1d(channels // 4, channels, 1),
                nn.Sigmoid()
            ) for _ in range(levels - 1)
        ])

    def forward(self, yl, yh):
        """
        跨尺度信息融合
        输入:
            yl: [B, C, L]  近似系数
            yh: list of [B, C, L] 细节系数
        输出:
            fused: [B, C, L] 融合后的近似系数
            enhanced: list of [B, C, L] 增强后的细节系数
        """
        # 从深层到浅层处理
        enhanced = []
        current = yl

        for i in range(self.levels - 1, -1, -1):
            # 当前尺度细节系数
            if i < len(yh):
                detail = yh[i]

                # 注意力机制增强细节
                if i < len(self.attentions):
                    attn = self.attentions[i](current)
                    detail = detail * attn + detail

                # 门控融合
                gate = self.gates[i](current)
                fused = current + gate * detail

                # 更新当前尺度
                current = fused
                enhanced.append(detail)
            else:
                enhanced.append(current)

        # 反转列表顺序 (浅层到深层)
        enhanced.reverse()

        return current, enhanced


# ====================== 自适应求解器封装 ======================
class AdaptiveSolver:
    """自适应ODE求解器封装"""

    def __init__(self, rtol=1e-6, atol=1e-8):
        self.rtol = rtol
        self.atol = atol

    def solve(self, func, y0, t, constants=None, method='dopri5'):
        """解ODE方程"""

        # 添加常量参数
        def ode_func(t, y):
            return func(t, y, **constants) if constants else func(t, y)

        return odeint(
            ode_func,
            y0,
            t,
            method=method,
            rtol=self.rtol,
            atol=self.atol
        )[-1]  # 只返回最终状态


# ====================== 组合创新: DyWAN-Pro + W-NDE-Pro + CSF ======================
class AdvancedWaveletDecomp(nn.Module):
    def __init__(self, channel, level=3, time_points=8, filter_length=8, dim=64, ode_solver='dopri5', ode_step=0.1):
        super().__init__()
        self.channel = channel
        self.level = level
        self.filter_length = filter_length
        self.time_points = time_points
        self.dim = dim

        # 动态小波生成器
        self.dywan = DyWAN_Pro(channel, filter_length, hidden_dim=dim)

        # 为每个尺度创建独立的ODE系统
        self.scale_odes = nn.ModuleList([
            ScaleAwareODE(channel, dim, time_dim=16)
            for _ in range(level + 1)
        ])

        # 自适应步长求解器
        self.ode_solver = AdaptiveSolver(rtol=1e-6, atol=1e-8)

        # 跨尺度交互模块
        self.cross_scale_fusion = CrossScaleFusion(channel, level)

        # 自适应时间点 (训练时随机，推理时均匀)
        self.register_buffer('train_time_points', torch.sort(torch.rand(time_points))[0])
        self.register_buffer('eval_time_points', torch.linspace(0, 1, time_points))

        self.ode_method = ode_solver
        self.ode_step = ode_step

    def adaptive_time_grid(self, signal):
        """基于信号局部特征生成时间网格"""
        B, C, L = signal.shape

        # 计算梯度 (长度变为 L-1)
        grad = torch.abs(torch.diff(signal, dim=-1))

        # 计算梯度强度 (长度变为 L-1)
        intensity = F.avg_pool1d(grad, kernel_size=5, padding=2)

        # 归一化到[0.1, 1.0]
        min_val = intensity.min()
        max_val = intensity.max()
        if max_val - min_val < 1e-8:
            t_density = torch.ones_like(intensity) * 0.5
        else:
            t_density = (intensity - min_val) / (max_val - min_val) * 0.9 + 0.1

        # 创建时间网格 (长度与 intensity 相同: L-1)
        time_grid = torch.zeros_like(t_density)

        # 只迭代到 L-2 避免越界
        for i in range(1, t_density.size(-1)):
            time_grid[..., i] = time_grid[..., i - 1] + t_density[..., i - 1]

        # 归一化到[0, 1]
        max_time = time_grid.max()
        if max_time > 0:
            time_grid = time_grid / max_time

        # 选择关键时间点 (在整个时间网格上均匀采样)
        indices = torch.linspace(0, time_grid.size(-1) - 1, self.time_points).long()
        selected_points = time_grid[..., indices].mean(dim=(0, 1))  # 取批次和通道的平均

        return selected_points.to(signal.device)

    def apply_energy_constraint(self, input_coeff, output_coeff):
        """应用能量守恒约束"""
        # 计算输入输出能量
        energy_in = torch.mean(input_coeff ** 2, dim=(1, 2), keepdim=True)
        energy_out = torch.mean(output_coeff ** 2, dim=(1, 2), keepdim=True)

        # 调整输出以匹配能量
        scale_factor = torch.sqrt(energy_in / (energy_out + 1e-8))
        return output_coeff * scale_factor

    def forward(self, x):
        B, C, L = x.shape
        filters = []
        coeffs = [x]  # level-0 近似

        # 多级分解
        approx = x
        for i in range(self.level):
            lo_f, hi_f, ortho_loss = self.dywan(approx)
            filters.append((lo_f, hi_f))

            # 边界处理：对称填充
            pad = (self.filter_length - 1) // 2
            ap_pad = F.pad(approx, (pad, pad), mode='replicate')

            # 准备卷积核
            lo_kernel = lo_f.unsqueeze(1).repeat(1, C, 1)
            hi_kernel = hi_f.unsqueeze(1).repeat(1, C, 1)

            # 卷积分解
            new_approx = F.conv1d(
                ap_pad.view(1, B * C, -1),
                lo_kernel.view(B * C, 1, self.filter_length),
                groups=B * C,
                padding=0
            ).view(B, C, -1)

            detail = F.conv1d(
                ap_pad.view(1, B * C, -1),
                hi_kernel.view(B * C, 1, self.filter_length),
                groups=B * C,
                padding=0
            ).view(B, C, -1)

            # 裁剪到原始长度
            new_approx = new_approx[..., :approx.size(2)]
            detail = detail[..., :approx.size(2)]

            coeffs.append(detail)
            approx = new_approx

        coeffs[0] = approx  # 最深层近似

        # 各尺度独立演化 + 物理约束
        evolved_coeffs = []
        for i, (coeff, ode) in enumerate(zip(coeffs, self.scale_odes)):
            # 自适应时间积分
            t_points = self.adaptive_time_grid(coeff)

            # 带物理约束的ODE演化
            evolved = self.ode_solver.solve(
                func=ode,
                y0=coeff,
                t=t_points,
                constants={'scale': i},
                method=self.ode_method
            )

            # 添加能量守恒约束
            evolved = self.apply_energy_constraint(coeff, evolved)
            evolved_coeffs.append(evolved)

        # 跨尺度融合 (保持小波结构)
        yl, yh = self.cross_scale_fusion(evolved_coeffs[0], evolved_coeffs[1:])

        return yl, yh, {"ortho_loss": ortho_loss}, filters

    def reconstruct(self, yl, yh, filters):
        B, C, L = yl.shape
        # 多级重构 (从深层到浅层)
        recon = yl
        for i in range(self.level - 1, -1, -1):
            lo_f, hi_f = filters[i]
            detail = yh[i]

            # 边界处理
            pad = (self.filter_length - 1) // 2
            rec_pad = F.pad(recon, (pad, pad), mode='replicate')
            det_pad = F.pad(detail, (pad, pad), mode='replicate')

            # 准备卷积核
            lo_kernel = lo_f.unsqueeze(1).repeat(1, C, 1)
            hi_kernel = hi_f.unsqueeze(1).repeat(1, C, 1)

            # 转置卷积重构
            recon_lo = F.conv_transpose1d(
                rec_pad.view(1, B * C, -1),
                lo_kernel.view(B * C, 1, self.filter_length),
                groups=B * C,
                padding=0
            ).view(B, C, -1)

            recon_hi = F.conv_transpose1d(
                det_pad.view(1, B * C, -1),
                hi_kernel.view(B * C, 1, self.filter_length),
                groups=B * C,
                padding=0
            ).view(B, C, -1)

            # 裁剪到原始长度
            recon_lo = recon_lo[..., pad:pad + L]
            recon_hi = recon_hi[..., pad:pad + L]

            # 合成
            recon = recon_lo + recon_hi

        return recon


# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    channels = 3
    seq_len = 128
    levels = 3

    # 创建模型
    model = AdvancedWaveletDecomp(channel=channels, level=levels, dim=64)

    # 创建测试数据
    x = torch.randn(batch_size, channels, seq_len)

    # 前向传播
    yl, yh, losses, filters = model(x)

    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"近似系数形状: {yl.shape}")
    for i, d in enumerate(yh):
        print(f"细节系数 {i + 1} 形状: {d.shape}")

    # 重建信号
    recon = model.reconstruct(yl, yh, filters)
    print(f"重建信号形状: {recon.shape}")

    # 打印损失
    print(f"正交性损失: {losses['ortho_loss'].item():.6f}")

    # 测试重建误差
    reconstruction_error = F.mse_loss(x, recon).item()
    print(f"重建MSE: {reconstruction_error:.6f}")