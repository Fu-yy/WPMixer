import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
import pywt
from einops import rearrange
from pytorch_wavelets import DWT1DForward, DWT1DInverse


# ====================== 基础模块 ======================
class RevIN(nn.Module):
    """可逆实例归一化层"""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=-1, keepdim=True)
            self.std = torch.std(x, dim=-1, keepdim=True)
            x = (x - self.mean) / (self.std + self.eps)
            return x * self.gamma + self.beta
        elif mode == 'denorm':
            return (x - self.beta) / self.gamma * (self.std + self.eps) + self.mean


# ====================== 创新点1: 动态小波基自适应网络 (DyWAN) ======================
class DyWAN(nn.Module):
    def __init__(self, channel, filter_length=8, hidden_dim=32):
        super().__init__()
        self.filter_length = filter_length
        # 统计特征提取
        self.stat_net = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channel, hidden_dim),
            nn.ReLU()
        )
        # 小波生成器
        self.wavelet_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, filter_length * 2)  # 低通+高通
        )
        # 正交性约束初始化
        self.ortho_loss = 0.0

    def forward(self, x):
        # x: [batch, channel, seq_len]
        stat_feat = self.stat_net(x)  # [batch, hidden]
        filters = self.wavelet_generator(stat_feat)  # [batch, 2*filter_length]
        lo_f = filters[:, :self.filter_length]  # [batch, filter_length]
        hi_f = filters[:, self.filter_length:]  # [batch, filter_length]

        # 添加正交性约束到损失（在训练时计算）
        self.ortho_loss = self._orthogonality_constraint(lo_f, hi_f)

        return lo_f, hi_f

    def _orthogonality_constraint(self, lo_f, hi_f):
        """计算滤波器组的正交性约束"""
        batch_size = lo_f.shape[0]
        loss = 0.0

        for i in range(batch_size):
            # 低通滤波器正交条件
            L = lo_f[i] / torch.norm(lo_f[i])
            for shift in range(1, self.filter_length, 2):
                shifted = torch.roll(L, shifts=shift)
                loss += torch.abs(torch.dot(L, shifted))

            # 高低通滤波器正交条件
            H = hi_f[i] / torch.norm(hi_f[i])
            loss += torch.abs(torch.dot(L, H))

            # 单位能量条件
            loss += torch.abs(torch.dot(L, L) - 1)
            loss += torch.abs(torch.dot(H, H) - 1)

        return loss / batch_size

    def reconstruct_wavelet(self, lo_f, hi_f):
        """可视化生成的小波基"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        # 低通滤波器
        plt.subplot(1, 2, 1)
        plt.plot(lo_f[0].detach().cpu().numpy(), label='Lo-Filter')
        plt.title(f"Low-pass Filter (Energy: {lo_f[0].pow(2).sum().item():.2f})")
        plt.grid(True)

        # 高通滤波器
        plt.subplot(1, 2, 2)
        plt.plot(hi_f[0].detach().cpu().numpy(), label='Hi-Filter')
        plt.title(f"High-pass Filter (Energy: {hi_f[0].pow(2).sum().item():.2f})")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('dynamic_wavelet.png')
        plt.close()


# ====================== 创新点2: 小波-神经微分方程融合 (W-NDE) ======================
class WaveletODE(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, time_dim=16):
        """
        完全重构的ODE函数

        参数:
            in_channels: 输入通道数
            hidden_dim: 隐藏层维度
            time_dim: 时间嵌入维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 核心ODE函数网络 - 保持输入输出维度一致
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, 3, padding=1)
        )

        # 时间条件层
        self.time_condition = nn.Linear(time_dim, hidden_dim * 2)

    def forward(self, t, x):
        """
        ODE函数前向传播

        参数:
            t: 时间点 (标量)
            x: 输入状态 [batch, in_channels, seq_len]

        返回:
            输出状态 [batch, in_channels, seq_len]
        """
        batch_size, _, seq_len = x.shape

        # 时间嵌入 [batch, time_dim]
        t_embed = self.time_embed(t.view(-1, 1))

        # 时间条件参数 [batch, hidden_dim * 2]
        t_params = self.time_condition(t_embed)
        gamma, beta = t_params.chunk(2, dim=1)  # 各为 [batch, hidden_dim]

        # 应用时间条件 (FiLM技术)
        gamma = gamma.unsqueeze(-1).expand(batch_size, -1, seq_len)
        beta = beta.unsqueeze(-1).expand(batch_size, -1, seq_len)

        # 通过核心网络
        h = self.net[0](x)  # 第一层卷积

        # 应用时间条件到中间层
        h = h * (1 + gamma) + beta  # 自适应缩放和偏移
        h = self.net[1](h)  # 激活函数

        # 剩余层
        for layer in self.net[2:]:
            h = layer(h)

        return h

    def energy_conservation_loss(self, x, dx_dt):
        """改进的能量守恒约束 - 使用L2范数"""
        # 计算输入能量
        energy_x = torch.norm(x, p=2, dim=(1, 2)) ** 2

        # 计算导数能量
        energy_dx = torch.norm(dx_dt, p=2, dim=(1, 2)) ** 2

        # 能量变化率应守恒
        return F.mse_loss(energy_x, energy_dx)

# ====================== 创新点3: 时变深度分解 (TV-D²) ======================
class AdaptiveGating(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, kernel_size, padding=kernel_size // 2)
        self.threshold = nn.Parameter(torch.tensor(0.5))  # 可学习阈值

    def forward(self, coeff):
        # coeff: [batch, channel, seq]
        importance = torch.sigmoid(self.conv(coeff))  # [batch, 1, seq]
        global_importance = importance.mean(dim=[1, 2])  # [batch]
        return global_importance > self.threshold  # 是否继续分解


class CrossLevelMixer(nn.Module):
    def __init__(self, channel, levels, reduction=4):
        super().__init__()
        self.levels = levels
        # 跨尺度交互模块
        self.mixer = nn.Sequential(
            nn.Conv1d(channel * (levels + 1), channel * reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel * reduction, channel, 1)
        )

    def forward(self, approx, details):
        # approx: [batch, channel, seq]
        # details: list of [batch, channel, seq]
        all_coeffs = [approx] + details
        mixed = self.mixer(torch.cat(all_coeffs, dim=1))
        return mixed


# ====================== 创新点4: 复小波域学习 (QCW-Mixer) ======================
class ComplexConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.real_conv = nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2)
        self.imag_conv = nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2)

    def forward(self, x_real, x_imag):
        real_out = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag_out = self.real_conv(x_imag) + self.imag_conv(x_real)
        return real_out, imag_out


class ComplexMixer(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net_real = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.net_imag = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x_real, x_imag):
        real_out = self.net_real(x_real) - self.net_imag(x_imag)
        imag_out = self.net_real(x_imag) + self.net_imag(x_real)
        return real_out, imag_out


# ====================== 创新点5: 可微分小波包熵剪枝 (DWPE) ======================
class DifferentiableEntropy(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, coeff):
        # coeff: [batch, channel, seq]
        energy = coeff.pow(2) + self.eps
        prob = energy / energy.sum(dim=-1, keepdim=True)
        entropy = - (prob * torch.log(prob)).sum(dim=-1)
        return entropy.mean(dim=1)  # [batch]


class GumbelTreeSelector(nn.Module):
    def __init__(self, num_nodes, temperature=0.5):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_nodes))
        self.temperature = temperature

    def forward(self, entropies):
        # entropies: [batch, num_nodes]
        logits = self.logits.unsqueeze(0) * entropies  # 熵加权
        return F.gumbel_softmax(logits, tau=self.temperature, hard=True)


# ====================== 组合创新: DyWAN + W-NDE ======================
class DyWAN_WNDE_Decomp(nn.Module):
    def __init__(self, channel, level=3, time_points=10, filter_length=8):
        super().__init__()
        self.channel = channel
        self.level = level
        self.filter_length = filter_length
        self.time_points = torch.linspace(0, 1, time_points)

        # 动态小波生成器
        self.dywan = DyWAN(channel, filter_length)

        # 神经微分方程 - 注意输入维度为3*channel
        self.w_ode = WaveletODE(in_channels=3 * channel)

        # 计算输出长度调整量
        self.output_pad = (filter_length - 1) // 2

    def reconstruct(self, yl, yh):
        """从系数重构信号（同时重构低高频）"""
        # 拼接 ODE 输入并反向求解（保持不变）
        coeffs = [yl] + yh
        combined = torch.cat(coeffs, dim=1)
        rev = odeint(
            self.w_ode,
            combined,
            torch.flip(self.time_points, [0]).to(combined.device),
            method='dopri5',
            rtol=1e-4, atol=1e-6
        )
        initial = rev[-1]
        x_recon = initial[:, :self.channel]
        approx_recon = initial[:, self.channel:2 * self.channel]
        detail_recon = initial[:, 2 * self.channel:3 * self.channel]

        # 动态滤波器：取 batch 均值生成共享低、高通
        lo_f, hi_f = self.dywan(x_recon)  # [batch, filter_length]
        lo_shared = lo_f.mean(dim=0)  # [filter_length]
        hi_shared = hi_f.mean(dim=0)

        # reshape 为 conv_transpose1d 要求的 weight 形状
        # out_ch=channel, in_ch/groups=1, k=filter_length
        lo_w = lo_shared.view(1, 1, -1).repeat(self.channel, 1, 1)
        hi_w = hi_shared.view(1, 1, -1).repeat(self.channel, 1, 1)

        # 填充
        pad_total = self.filter_length - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # 低频 & 高频 反卷积
        lo_pad = F.pad(approx_recon, (pad_left, pad_right), mode='reflect')
        hi_pad = F.pad(detail_recon, (pad_left, pad_right), mode='reflect')
        recon_lo = F.conv_transpose1d(lo_pad, lo_w, groups=self.channel)
        recon_hi = F.conv_transpose1d(hi_pad, hi_w, groups=self.channel)

        # 合并并裁剪到原始长度
        recon = (recon_lo + recon_hi)[:, :, :x_recon.size(2)]
        return recon

    def forward(self, x):
        # x: [batch, channel, seq_len]
        batch, channel, seq_len = x.shape
        # 动态生成小波基
        lo_f, hi_f = self.dywan(x)

        # 取 batch 均值，做共享滤波器
        lo_shared = lo_f.mean(0)  # [filter_length]
        hi_shared = hi_f.mean(0)
        lo_w = lo_shared.view(1, 1, -1).repeat(channel, 1, 1)
        hi_w = hi_shared.view(1, 1, -1).repeat(channel, 1, 1)

        pad_total = self.filter_length - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_pad = F.pad(x, (pad_left, pad_right), mode='reflect')

        approx = F.conv1d(x_pad, lo_w, groups=channel)[:, :, :seq_len]
        detail = F.conv1d(x_pad, hi_w, groups=channel)[:, :, :seq_len]

        combined = torch.cat([x, approx, detail], dim=1)
        coeffs = odeint(self.w_ode, combined, self.time_points.to(x.device),
                        method='dopri5', rtol=1e-4, atol=1e-6)
        final_state = coeffs[-1]

        # 重构分解系数
        yl = final_state[:, :self.channel]  # 最终近似
        yh = []

        # 提取多级细节系数
        for i in range(self.level):
            start_idx = self.channel + i * self.channel
            end_idx = start_idx + self.channel
            if end_idx <= final_state.size(1):
                yh.append(final_state[:, start_idx:end_idx])
            else:
                # 如果超出范围，添加零填充
                yh.append(torch.zeros_like(yl))

        # 返回损失用于正则化
        losses = {
            "ortho_loss": self.dywan.ortho_loss,
            "energy_loss": self.w_ode.energy_conservation_loss(combined, coeffs[-1])
        }
        return yl, yh, losses

# ====================== 完整分解模块 ======================
class AdvancedDecomposition(nn.Module):
    def __init__(self,
                 channel,
                 level=3,
                 mode='dywan_nde',  # 'dywan', 'nde', 'tv', 'complex', 'dwpe', 'dywan_nde'
                 filter_length=8,
                 use_complex=False):
        super().__init__()
        self.channel = channel
        self.level = level
        self.mode = mode
        self.use_complex = use_complex

        # 根据模式选择分解方法
        if mode == 'dywan_nde':
            self.decomp = DyWAN_WNDE_Decomp(channel, level, filter_length=filter_length)
        elif mode == 'tv':
            self.gating = nn.ModuleList([AdaptiveGating(channel) for _ in range(level)])
            self.cross_mixer = CrossLevelMixer(channel, level)
        elif mode == 'dwpe':
            self.entropy = DifferentiableEntropy()
            self.selector = GumbelTreeSelector(num_nodes=2 ** level - 1)

        # 复小波处理
        if use_complex:
            self.complex_mixer = ComplexMixer(channel)

        # 传统小波作为备选
        self.dwt = DWT1DForward(wave='db4', J=level)

    def forward(self, x):
        if self.mode == 'dywan_nde':
            return self.decomp(x)

        elif self.mode == 'tv':
            return self._time_varying_decomp(x)

        elif self.mode == 'dwpe':
            return self._entropy_pruning_decomp(x)

        else:
            # 传统小波分解
            return self.dwt(x)

    def _time_varying_decomp(self, x):
        """时变深度分解"""
        approx = x
        details = []
        active_levels = []

        for i in range(self.level):
            # 单级分解
            approx, detail = self._single_level_dwt(approx)

            # 应用门控
            if self.gating[i](approx):
                details.append(detail)
                active_levels.append(i)
            else:
                details.append(detail)
                break

        # 跨尺度融合
        if len(active_levels) > 1:
            approx = self.cross_mixer(approx, [details[i] for i in active_levels])

        return approx, details, {"active_levels": len(active_levels)}

    def _entropy_pruning_decomp(self, x):
        """可微分小波包熵剪枝"""
        # 构建完整小波包树
        full_tree = self._build_full_tree(x)

        # 计算节点熵
        entropies = torch.stack([self.entropy(node) for node in full_tree], dim=1)

        # 选择保留节点
        masks = self.selector(entropies)

        # 应用剪枝
        pruned_tree = [node * masks[:, i:i + 1] for i, node in enumerate(full_tree)]

        # 重构
        approx = self._reconstruct_tree(pruned_tree)
        return approx, pruned_tree, {"retained_nodes": masks.sum(dim=1).mean()}

    def _build_full_tree(self, x):
        """构建完整小波包树（简化实现）"""
        # 实际实现需要使用小波包变换
        tree = [x]
        current_level = [x]

        for _ in range(self.level):
            next_level = []
            for node in current_level:
                a, d = self._single_level_dwt(node)
                next_level.extend([a, d])
            tree.extend(next_level)
            current_level = next_level

        return tree


# ====================== 验证与测试模块 ======================
def run_dywan():
    """测试动态小波基生成"""
    dywan = DyWAN(channel=8)
    x = torch.randn(32, 8, 100)
    lo_f, hi_f = dywan(x)
    print(f"DyWAN 输出: lo_f={lo_f.shape}, hi_f={hi_f.shape}")
    print(f"正交损失: {dywan.ortho_loss.item():.4f}")
    dywan.reconstruct_wavelet(lo_f, hi_f)
    print("小波基可视化已保存到 dynamic_wavelet.png")


def run_nde():
    """测试神经微分方程（终极修复版）"""
    # 设置输入通道数
    in_channels = 8

    # 创建ODE函数
    ode_func = WaveletODE(in_channels=in_channels)

    # 创建测试数据
    x = torch.randn(32, in_channels, 100)

    # 创建时间点张量
    t = torch.tensor([0.0, 0.5, 1.0])

    # 确保设备一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    ode_func = ode_func.to(device)
    x = x.to(device)
    t = t.to(device)

    # 解ODE
    try:
        coeffs = odeint(
            ode_func,
            x,
            t,
            method='dopri5',
            rtol=1e-4,
            atol=1e-6
        )
        print(f"ODE求解成功! 输出: {len(coeffs)} 时间步, 形状={coeffs[0].shape}")

        # 计算能量守恒损失
        x0 = coeffs[0]
        x_final = coeffs[-1]

        # 计算导数近似值
        dx_dt = (x_final - x0) / (t[-1] - t[0] + 1e-7)

        loss = ode_func.energy_conservation_loss(x0, dx_dt)
        print(f"能量守恒损失: {loss.item():.4f}")

    except Exception as e:
        print(f"ODE求解失败: {str(e)}")
        # 调试信息
        print("调试信息:")
        print(f"输入形状: {x.shape}")
        print(f"时间点: {t}")
        print(f"模型结构: {ode_func}")

        # 测试单步前向
        try:
            test_output = ode_func(t[0], x)
            print(f"单步前向成功! 输出形状: {test_output.shape}")
        except Exception as e2:
            print(f"单步前向失败: {str(e2)}")


def run_combined():
    """测试组合模型（修复版）"""
    decomp = DyWAN_WNDE_Decomp(channel=8, level=3)
    x = torch.randn(32, 8, 100)
    try:
        yl, yh, losses = decomp(x)
        print(f"组合分解成功!")
        print(f"近似系数形状: {yl.shape}")
        print(f"细节系数: {[d.shape for d in yh]}")
        print(f"正交损失: {losses['ortho_loss'].item():.4f}")
        print(f"能量损失: {losses['energy_loss'].item():.4f}")

        # 验证重构精度
        recon_loss = F.mse_loss(
            decomp.reconstruct(yl, yh),
            x
        )
        print(f"重构损失: {recon_loss.item():.6f}")

    except Exception as e:
        print(f"组合分解失败: {str(e)}")
        # 调试信息
        print(f"输入形状: {x.shape}")
        print(f"滤波器长度: {decomp.filter_length}")

        # 测试小波卷积部分
        lo_f, hi_f = decomp.dywan(x)
        print(f"低通滤波器形状: {lo_f.shape}")
        print(f"高通滤波器形状: {hi_f.shape}")

        # 测试ODE部分
        test_input = torch.randn(32, 24, 100).to(x.device)
        test_output = decomp.w_ode(torch.tensor(0.0), test_input)
        print(f"ODE单步输出形状: {test_output.shape}")

def run_complex():
    """测试复数混合"""
    mixer = ComplexMixer(dim=64)
    x_real = torch.randn(32, 128, 64)
    x_imag = torch.randn(32, 128, 64)
    y_real, y_imag = mixer(x_real, x_imag)
    print(f"复数混合输出: real={y_real.shape}, imag={y_imag.shape}")


def reconstruction_error_test(module, input_shape=(32, 8, 100)):
    """重构误差测试"""
    x = torch.randn(*input_shape)
    yl, yh, _ = module(x)
    recon = module.decomp.reconstruct(yl, yh)
    error = F.mse_loss(x, recon).item()
    print(f"重构MSE: {error:.6f}")
    assert error < 1e-4, "重构误差过大!"
    return error


def energy_conservation_test(module, input_shape=(32, 8, 100)):
    """能量守恒测试"""
    x = torch.randn(*input_shape)
    yl, yh, _ = module(x)

    # 计算输入能量
    input_energy = x.pow(2).sum().item()

    # 计算系数能量
    coeff_energy = yl.pow(2).sum().item()
    for d in yh:
        coeff_energy += d.pow(2).sum().item()

    error = abs(input_energy - coeff_energy) / input_energy
    print(f"能量守恒误差: {error * 100:.2f}%")
    assert error < 0.01, "能量守恒失败!"
    return error


# ====================== 主执行与验证 ======================
if __name__ == "__main__":
    print("=" * 50)
    print("测试动态小波基 (DyWAN)")
    run_dywan()

    print("\n" + "=" * 50)
    print("测试神经微分方程 (W-NDE)")
    run_nde()

    print("\n" + "=" * 50)
    print("测试组合模型 (DyWAN + W-NDE)")
    run_combined()

    print("\n" + "=" * 50)
    print("测试复数混合 (QCW-Mixer)")
    run_complex()

    print("\n" + "=" * 50)
    print("验证重构误差")
    decomp = AdvancedDecomposition(channel=8, mode='dywan_nde')
    reconstruction_error_test(decomp)

    print("\n" + "=" * 50)
    print("验证能量守恒")
    energy_conservation_test(decomp)




    # ------------------
    # 初始化组合创新分解器
    decomp = AdvancedDecomposition(
        channel=8,
        mode='dywan_nde',
        level=3,
        filter_length=8
    )

    # 执行分解
    x = torch.randn(32, 8, 100)  # 输入数据
    yl, yh, losses = decomp(x)

    # 验证关键指标
    print(f"近似系数: {yl.shape}")
    print(f"细节系数: {len(yh)} levels")
    print(f"正交损失: {losses['ortho_loss'].item():.4f}")
    print(f"能量损失: {losses['energy_loss'].item():.4f}")

    # 运行验证套件
    reconstruction_error_test(decomp)
    energy_conservation_test(decomp)