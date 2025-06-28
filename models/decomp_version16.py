import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
from einops import rearrange
import math
import matplotlib.pyplot as plt

# ====================== 理论基础 ======================
"""
科学依据：
1. 动态小波基(DyWAN)：通过统计特征生成自适应小波基，解决了传统小波基固定不变的问题
   - 理论依据：小波基的适应性原理(Mallat, 2009) + 深度学习特征提取
   - 创新点：正交性约束保证完美重构，频带自适应提升分解质量

2. 小波神经微分方程(W-NDE)：将小波系数视为连续动力系统
   - 理论依据：神经ODE(Chen et al., 2018) + 信号能量守恒原理
   - 创新点：时间感知的微分方程建模小波系数演化，功率守恒约束保持物理意义

3. 跨尺度融合机制(CSF)：建立不同尺度间的信息交互
   - 理论依据：多分辨率分析框架 + 注意力机制
   - 创新点：门控机制控制尺度间信息流，残差结构保持信息完整性
"""


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


# ====================== 创新点2: 增强型小波神经微分方程 (W-NDE Pro) ======================
class WaveletODE_Pro(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, time_dim=16):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 时间嵌入层（带周期性）
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
        )

        # 修改网络结构：明确区分卷积层和激活层
        self.conv = nn.Conv1d(in_channels, in_channels, 3, padding=1, padding_mode='replicate')
        # self.conv = nn.Linear(in_channels, in_channels)
        # self.conv = nn.Linear(96, 96)


        self.conv1 = nn.Conv1d(in_channels, hidden_dim, 3, padding=1, padding_mode='replicate')
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='replicate')
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(hidden_dim, in_channels, 3, padding=1, padding_mode='replicate')
        self.tanh = nn.Tanh()

        # 为每个卷积层创建独立的条件变换
        self.time_condition = nn.Linear(time_dim, in_channels * 2)

        self.time_condition1 = nn.Linear(time_dim, hidden_dim * 2)
        self.time_condition2 = nn.Linear(time_dim, hidden_dim * 2)
        self.time_condition3 = nn.Linear(time_dim, in_channels * 2)
        self.damping = nn.Parameter(torch.tensor(0.01))

    def forward(self, t, x):
        # 时间嵌入
        t_embed = self.time_embed(t.view(-1, 1))  # [batch, time_dim]

        # 第一卷积层 + 条件 + 激活
        h = self.conv(x)
        gamma1, beta1 = self.time_condition(t_embed).chunk(2, dim=1)
        # 扩展条件参数以匹配特征图形状
        gamma1 = gamma1.view(-1, self.in_channels, 1)
        beta1 = beta1.view(-1, self.in_channels, 1)
        h = h * (1 + gamma1) + beta1  # 条件注入
        # h = self.act1(h)
        # # 第一卷积层 + 条件 + 激活
        # h = self.conv1(x)
        # gamma1, beta1 = self.time_condition1(t_embed).chunk(2, dim=1)
        # # 扩展条件参数以匹配特征图形状
        # gamma1 = gamma1.view(-1, self.hidden_dim, 1)
        # beta1 = beta1.view(-1, self.hidden_dim, 1)
        # h = h * (1 + gamma1) + beta1  # 条件注入
        # h = self.act1(h)

        # # 第二卷积层 + 条件 + 激活
        # h = self.conv2(h)
        # gamma2, beta2 = self.time_condition2(t_embed).chunk(2, dim=1)
        # gamma2 = gamma2.view(-1, self.hidden_dim, 1)
        # beta2 = beta2.view(-1, self.hidden_dim, 1)
        # h = h * (1 + gamma2) + beta2
        # h = self.act2(h)
        #
        # # 第三卷积层 + 条件（无激活）
        # h = self.conv3(h)
        # gamma3, beta3 = self.time_condition3(t_embed).chunk(2, dim=1)
        # gamma3 = gamma3.view(-1, self.in_channels, 1)
        # beta3 = beta3.view(-1, self.in_channels, 1)
        # h = h * (1 + gamma3) + beta3
        #
        # # 最终激活
        h = self.tanh(h)

        # 阻尼项
        # return h - self.damping * x ** 3
        return h - self.damping * x
    # 修正能量计算（考虑时间积分）
    def power_conservation_loss(self, x, dx_dt):
        # 输入能量变化率
        dE_input = 2 * torch.mean(x * dx_dt, dim=(1, 2))

        # 系统耗散功率
        dissipation = torch.mean(dx_dt ** 2, dim=(1, 2))

        # 应满足 dE/dt = -dissipation
        return F.mse_loss(dE_input, -dissipation)



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


# ====================== 组合创新: DyWAN-Pro + W-NDE-Pro + CSF ======================
class AdvancedWaveletDecomp(nn.Module):
    def __init__(self, channel, level=3, time_points=8, filter_length=8,dim=512,ode_solver='dopri5', ode_step=0.1):
        super().__init__()
        self.channel = channel
        self.level = level
        self.filter_length = filter_length

        # 动态小波生成器
        self.dywan = DyWAN_Pro(channel, filter_length,hidden_dim=dim)

        # 神经微分方程
        self.w_ode = WaveletODE_Pro(in_channels=(1 + level) * channel,hidden_dim=dim)

        # 跨尺度融合
        self.csf = CrossScaleFusion(channel, level)

        # 自适应时间点 (训练时随机，推理时均匀)
        self.train_time_points = torch.sort(torch.rand(time_points))[0]
        self.eval_time_points = torch.linspace(0, 1, time_points)

        self.ode_solver = ode_solver
        self.ode_step = ode_step

    # 替换conv1d实现
    def efficient_conv(self,x, kernel, groups):
        return F.conv1d(
            x,
            kernel.unsqueeze(0),  # [1, groups, filter_length]
            groups=groups,
            padding=0
        )
    def forward(self, x):
        # 存储滤波器组和分解系数
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
            # ap_pad = F.pad(approx, (pad, pad), mode='replicate')  # [B, C, L+2*pad]
            ap_pad = F.pad(approx, (pad, pad), mode='replicate')  # [B, C, L+2*pad]

            # 准备卷积核
            # 将滤波器扩展为与输入通道匹配的形状 [B, C, filter_length]
            lo_kernel = lo_f.unsqueeze(1).repeat(1, C, 1)  # [B, C, filter_length]
            hi_kernel = hi_f.unsqueeze(1).repeat(1, C, 1)  # [B, C, filter_length]

            # 卷积分解 - 使用分组卷积确保每个通道有自己的滤波器
            # 注意：这里使用groups=B*C，因为每个样本的每个通道都有独立的滤波器

            new_approx = F.conv1d(
                ap_pad.view(1, B * C, -1),  # 将批量合并为单个通道维度 [1, B*C, L+2*pad]
                lo_kernel.view(B * C, 1, self.filter_length),  # [B*C, 1, filter_length]
                groups=B * C,
                padding=0
            ).view(B, C, -1)  # 恢复形状 [B, C, L]




            detail = F.conv1d(
                ap_pad.view(1, B * C, -1),  # [1, B*C, L+2*pad]
                hi_kernel.view(B * C, 1, self.filter_length),  # [B*C, 1, filter_length]
                groups=B * C,
                padding=0
            ).view(B, C, -1)  # [B, C, L]

            # 裁剪到原始长度
            new_approx = new_approx[..., :approx.size(2)]
            detail = detail[..., :approx.size(2)]

            coeffs.append(detail)
            approx = new_approx

        coeffs[0] = approx  # 最深层近似

        # 跨尺度融合
        approx_fused, details_fused = self.csf(approx, coeffs[1:])
        coeffs_fused = [approx_fused] + details_fused

        # 拼接系数 [B, (1+level)*C, L]
        combined = torch.cat(coeffs_fused, dim=1)

        # ODE演化 (自适应时间点)
        time_points = self.train_time_points if self.training else self.eval_time_points
        # ode_out = odeint(
        #     self.w_ode,
        #     combined,
        #     time_points.to(x.device),
        #     method=self.ode_solver,
        #     rtol=1e-5,
        #     atol=1e-7,
        #     options={'step_size': self.ode_step} if 'fixed' in self.ode_solver else {}
        # )[-1]  # 取最终状态
        # # 修改forward中的odeint调用
        ode_out = combined



        # ode_out = odeint(
        #     self.w_ode,
        #     combined,
        #     time_points.to(x.device),
        #     method=self.ode_solver,
        #     rtol=1e-5,
        #     atol=1e-6,
        #     options={'step_size': self.ode_step} if 'fixed' in self.ode_solver else {}
        # )[-1]  # 取最终状态
        # ode_out = ode_out + combined
        # 分解输出
        yl = ode_out[:, :self.channel, :]
        yh = []
        for i in range(self.level):
            start = self.channel * (i + 1)
            end = start + self.channel
            yh.append(ode_out[:, start:end, :])

        # 计算功率守恒损失
        with torch.no_grad():
            dx_dt = (ode_out - combined) / (time_points[-1] - time_points[0] + 1e-7)
            energy_loss = self.w_ode.power_conservation_loss(combined, dx_dt)

        return yl, yh, {"ortho_loss": ortho_loss, "energy_loss": energy_loss}, filters

    def reconstruct(self, yl, yh, filters):
        B, C, L = yl.shape
        # 拼接系数
        coeffs = [yl] + yh
        combined = torch.cat(coeffs, dim=1)

        # 反向ODE演化
        time_points = self.train_time_points if self.training else self.eval_time_points
        rev_points = torch.flip(time_points, [0])
        ode_rev = combined
        # ode_rev = odeint(
        #     self.w_ode,
        #     combined,
        #     rev_points.to(yl.device),
        #     method=self.ode_solver,
        #     rtol=1e-5,
        #     atol=1e-6,
        #     options={'step_size': self.ode_step} if 'fixed' in self.ode_solver else {}
        # )[-1]

        # 拆解系数
        approx = ode_rev[:, :self.channel, :]
        details = [ode_rev[:, self.channel * (i + 1):self.channel * (i + 2), :]
                   for i in range(self.level)]

        # 多级重构 (从深层到浅层)
        recon = approx
        for i in range(self.level - 1, -1, -1):
            lo_f, hi_f = filters[i]

            # 边界处理
            pad = (self.filter_length - 1) // 2
            rec_pad = F.pad(recon, (pad, pad), mode='replicate')
            det_pad = F.pad(details[i], (pad, pad), mode='replicate')

            # 准备卷积核
            # 将滤波器扩展为与输入通道匹配的形状 [B, C, filter_length]
            lo_kernel = lo_f.unsqueeze(1).repeat(1, C, 1)  # [B, C, filter_length]
            hi_kernel = hi_f.unsqueeze(1).repeat(1, C, 1)  # [B, C, filter_length]

            # 转置卷积重构 - 使用分组卷积
            recon_lo = F.conv_transpose1d(
                rec_pad.view(1, B * C, -1),  # [1, B*C, L+2*pad]
                lo_kernel.view(B * C, 1, self.filter_length),  # [B*C, 1, filter_length]
                groups=B * C,
                padding=0
            ).view(B, C, -1)  # [B, C, L+filter_length-1]

            recon_hi = F.conv_transpose1d(
                det_pad.view(1, B * C, -1),  # [1, B*C, L+2*pad]
                hi_kernel.view(B * C, 1, self.filter_length),  # [B*C, 1, filter_length]
                groups=B * C,
                padding=0
            ).view(B, C, -1)  # [B, C, L+filter_length-1]
            # recon_lo = rec_pad
            # recon_hi = det_pad
            # 裁剪到原始长度
            recon_lo = recon_lo[..., pad:pad + L]
            recon_hi = recon_hi[..., pad:pad + L]

            # 合成
            recon = recon_lo + recon_hi

        return recon


# ====================== 重构网络 01 直接重构 ====================================
class DirectForecaster(nn.Module):
    def __init__(self, decomp, pred_len, levels=3, channels=8):
        super().__init__()
        self.decomp = decomp
        self.pred_len = pred_len

        # 多尺度特征处理器
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)  # 保留通道维度
            ) for _ in range(levels + 1)  # +1 包含yl
        ])

        # 跨尺度注意力融合
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(64 * (levels + 1), 256),
            nn.GELU(),
            nn.Linear(256, pred_len * channels),
            nn.Unflatten(1, (channels, pred_len))
        )

    def forward(self, x):
        # 分解获取多尺度特征
        yl, yh, losses, _ = self.decomp(x)
        all_scales = [yl] + yh

        # 各尺度独立处理
        scale_features = []
        for i, processor in enumerate(self.scale_processors):
            feat = processor(all_scales[i])  # [B, 64, 1]
            scale_features.append(feat.squeeze(-1))  # [B, 64]

        # 注意力融合
        features = torch.stack(scale_features, dim=1)  # [B, levels+1, 64]
        attn_out, _ = self.attention(features, features, features)

        # 展平预测
        fused = attn_out.flatten(1)  # [B, (levels+1)*64]
        return self.head(fused), losses
# ====================== 重构网络 02 系数演化预测 ====================================

# 利用ODE演化轨迹进行预测
class ODEForecaster(nn.Module):
    def __init__(self, decomp, pred_len):
        super().__init__()
        self.decomp = decomp
        self.pred_len = pred_len

        # 捕获ODE动态特征
        self.temporal_net = nn.GRU(
            input_size=decomp.channel * (decomp.level + 1),
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # 动态预测头
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, pred_len * decomp.channel)
        )

    def forward(self, x,yl,yh):
        # 获取完整ODE轨迹
        with torch.no_grad():
            _, _, _, filters = self.decomp(x)
            time_points = self.decomp.eval_time_points

        # 获取ODE演化全过程
        coeffs = torch.cat([yl] + yh, dim=1)
        full_trajectory = odeint(
            self.decomp.w_ode,
            coeffs,
            time_points.to(x.device),
            method='dopri5'
        )  # [T, B, C*(L+1), L]

        # 时间维度处理
        trajectory = rearrange(full_trajectory, "t b c l -> b t (c l)")

        # GRU处理时序动态
        output, _ = self.temporal_net(trajectory)

        # 预测未来
        last_state = output[:, -1, :]
        pred = self.head(last_state)
        return pred.reshape(-1, self.decomp.channel, self.pred_len)


# ====================== 验证测试 ======================
def run_dywan_pro():
    """测试增强型动态小波基"""
    dywan = DyWAN_Pro(channel=8)
    x = torch.randn(32, 8, 100)
    lo_f, hi_f, loss = dywan(x)
    print(f"DyWAN-Pro 输出: lo_f={lo_f.shape}, hi_f={hi_f.shape}")
    print(f"正则化损失: {loss.item():.6f}")

    # 可视化滤波器
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(lo_f[0].detach().cpu().numpy())
    plt.title("Dynamic Low-pass Filter")
    plt.grid(True)
    plt.subplot(122)
    plt.plot(hi_f[0].detach().cpu().numpy())
    plt.title("Dynamic High-pass Filter")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dywan_pro_filters.png')
    plt.close()


def run_nde_pro():
    """测试增强型神经微分方程"""
    ode_func = WaveletODE_Pro(in_channels=8)
    x = torch.randn(32, 8, 100)
    t = torch.tensor([0.0, 0.5, 1.0])

    # 解ODE
    coeffs = odeint(
        ode_func,
        x,
        t,
        method='dopri5',
        rtol=1e-5,
        atol=1e-7
    )
    print(f"ODE求解成功! 步数: {len(coeffs)}, 形状: {coeffs[0].shape}")

    # 计算功率守恒
    dx_dt = (coeffs[-1] - coeffs[0]) / (t[-1] - t[0])
    loss = ode_func.power_conservation_loss(coeffs[0], dx_dt)
    print(f"功率守恒损失: {loss.item():.6f}")


def run_full_decomp():

    batch = 32
    seqlen = 96
    nvar = 7
    x = torch.randn(batch, seqlen, nvar)
    """测试完整分解重构系统"""
    decomp = AdvancedWaveletDecomp(channel=nvar, level=2, filter_length=5)
    x = x.permute(0,2,1)  # 使用较小的batch size

    # 分解
    yl, yh, losses, filters = decomp(x)


    print(f"分解成功! 近似: {yl.shape}, 细节: {[d.shape for d in yh]}")
    print(f"正交损失: {losses['ortho_loss'].item():.6f}")
    print(f"能量损失: {losses['energy_loss'].item():.6f}")

    # 重构
    recon = decomp.reconstruct(yl, yh, filters)
    error = F.mse_loss(x, recon).item()
    print(f"重构误差: {error:.8f}")

    # 能量守恒验证
    input_energy = x.pow(2).sum().item()
    output_energy = yl.pow(2).sum().item() + sum(d.pow(2).sum().item() for d in yh)
    energy_error = abs(input_energy - output_energy) / input_energy
    print(f"能量守恒误差: {energy_error * 100:.4f}%")

    # 可视化原始信号和重构信号
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(x[0, 0].detach().cpu().numpy(), label='Original')
    plt.title("Original Signal")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(recon[0, 0].detach().cpu().numpy(), label='Reconstructed', color='orange')
    plt.title("Reconstructed Signal")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('signal_reconstruction.png')
    plt.close()


if __name__ == "__main__":
    print("=" * 50, "\n测试DyWAN-Pro")
    run_dywan_pro()

    print("\n" + "=" * 50, "\n测试W-NDE-Pro")
    run_nde_pro()

    print("\n" + "=" * 50, "\n测试完整分解重构系统")
    run_full_decomp()