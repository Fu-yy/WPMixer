import torch
import torch.nn as nn
import torchdiffeq as ode


class NeuralODEDecomp(nn.Module):
    def __init__(self, nvar=7, ode_hidden=64,
                 components=3, device='cuda'):
        super().__init__()
        self.nvar = nvar
        self.ode_hidden = ode_hidden
        self.components = components
        self.device = device

        # ODE函数网络
        self.ode_func = ODEFunc(nvar, ode_hidden, components)

        # 分量初始化网络
        self.init_net = nn.Sequential(
            nn.Linear(nvar, ode_hidden),
            nn.ReLU(),
            nn.Linear(ode_hidden, components * ode_hidden)
        )

        # 分量解码网络
        self.decode_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ode_hidden, ode_hidden),
                nn.ReLU(),
                nn.Linear(ode_hidden, nvar)
            ) for _ in range(components)
        ])

        # 时间点 (0到1的均匀分布)
        self.t_points = torch.linspace(0, 1, 96).to(device)

    def forward(self, x):
        """输入: (batch_size, seq_len, nvar)"""
        batch_size = x.shape[0]

        # 初始化分量状态
        init_state = self.init_net(x.mean(dim=1))
        init_state = init_state.view(batch_size, self.components, self.ode_hidden)

        # 求解ODE
        states = ode.odeint(self.ode_func, init_state, self.t_points)
        # states: (seq_len, batch, components, hidden)
        states = states.permute(1, 0, 2, 3)  # (batch, seq, comp, hidden)

        # 解码分量
        components = []
        for i in range(self.components):
            comp = self.decode_net[i](states[..., i, :])
            components.append(comp)

        # 第一个分量为趋势/低频
        trend = components[0]
        # 其余为高频/随机分量
        high_freqs = components[1:]

        return trend, high_freqs

    def inverse(self, trend, high_freqs):
        """重构信号: 分量求和"""
        recon = trend.clone()
        for comp in high_freqs:
            recon += comp
        return recon


class ODEFunc(nn.Module):
    """定义分量动力学"""

    def __init__(self, nvar, hidden_dim, components):
        super().__init__()
        self.nvar = nvar
        self.components = components

        # 动力学网络
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + components, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # 分量交互矩阵
        self.interaction = nn.Parameter(torch.eye(components))

        # 物理约束: 分量守恒
        self.conservation = nn.Linear(components, components, bias=False)
        nn.init.eye_(self.conservation.weight)
        self.conservation.weight.requires_grad = False

    def forward(self, t, state):
        """state: (batch, components, hidden)"""
        batch_size, comps, hidden = state.shape

        # 添加时间信息
        t_vec = torch.ones(batch_size, comps, 1, device=state.device) * t

        # 添加分量间相互作用
        interaction = self.interaction.unsqueeze(0).repeat(batch_size, 1, 1)
        state_comp = state.mean(dim=-1)  # (batch, comp)
        interaction_effect = torch.bmm(interaction, state_comp.unsqueeze(-1))

        # 物理约束: 分量守恒
        interaction_effect = self.conservation(interaction_effect.squeeze()).unsqueeze(-1)

        # 组合输入
        state_in = state.view(batch_size * comps, hidden)
        interaction_effect = interaction_effect.view(batch_size * comps, 1)
        t_vec = t_vec.view(batch_size * comps, 1)
        inputs = torch.cat([state_in, interaction_effect, t_vec], dim=1)

        # 计算导数
        d_state = self.net(inputs)
        d_state = d_state.view(batch_size, comps, hidden)

        # 确保能量守恒
        total_energy = (state ** 2).sum(dim=(1, 2))
        new_energy = ((state + d_state * 0.01) ** 2).sum(dim=(1, 2))
        energy_constraint = (new_energy - total_energy).mean() * 100

        # 添加能量约束作为正则化
        if self.training:
            self.energy_constraint = energy_constraint

        return d_state


# 使用示例
if __name__ == "__main__":
    decomp = NeuralODEDecomp(nvar=7, components=3).cuda()

    # 模拟输入 (32, 96, 7)
    x = torch.randn(32, 96, 7).cuda()

    # 分解
    trend, high_freqs = decomp(x)
    print(f"趋势分量形状: {trend.shape}")
    print(f"高频分量数量: {len(high_freqs)} | 第一个高频形状: {high_freqs[0].shape}")

    # 重构
    recon = decomp.inverse(trend, high_freqs)
    print(f"重构误差: {torch.abs(x - recon).mean().item():.4f}")