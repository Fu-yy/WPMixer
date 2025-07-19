import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicChebyKAN(nn.Module):
    """可适应谱分辨率的切比雪夫KAN层（时变多项式阶数）"""

    def __init__(self, input_dim, output_dim, base_degree, time_steps):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_degree = base_degree
        self.time_steps = time_steps

        # 时变多项式阶数参数化
        self.degree_params = nn.Parameter(torch.linspace(0, 1, time_steps))
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, base_degree + 3))
        nn.init.kaiming_uniform_(self.coeffs, a=5 ** 0.5)

        # 注册缓冲区
        self.register_buffer("cheby_range", torch.arange(0, base_degree + 3, 1, dtype=torch.float32))

        # 自适应截断参数
        self.epsilon = nn.Parameter(torch.tensor(1e-7))
        self.epsilon.requires_grad = False

    def get_degree(self, t):
        """动态调整多项式阶数（创新点1）"""
        scaled = 0.5 * (1 + torch.sin(2 * torch.pi * self.degree_params[t]))
        return int(self.base_degree + 2 * scaled.item())

    def forward(self, x, t):
        # 动态选择多项式阶数
        degree = self.get_degree(t)
        active_range = self.cheby_range[:degree + 1]

        # 安全变换
        t_val = torch.tanh(x).clamp(-1 + self.epsilon.item(), 1 - self.epsilon.item())
        theta = torch.acos(t_val)

        # 切比雪夫基函数计算
        cheb = torch.cos(theta.unsqueeze(-1) * active_range)

        # 选择激活系数
        active_coeffs = self.coeffs[..., :degree + 1]

        # 多项式展开
        y = torch.einsum("bnd,iod->bno", cheb, active_coeffs)
        return y


class DifferentiableKalmanCell(nn.Module):
    """可微分卡尔曼滤波单元（创新点2）"""

    def __init__(self, state_dim, obs_dim, base_degree, time_steps):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # 状态转移函数（切比雪夫网络）
        self.state_transfer = DynamicChebyKAN(state_dim, state_dim, base_degree, time_steps)

        # 观测模型
        self.obs_matrix = nn.Parameter(torch.eye(obs_dim, state_dim))
        nn.init.orthogonal_(self.obs_matrix)

        # 噪声协方差（对数空间参数化）
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim))
        self.log_R_diag = nn.Parameter(torch.zeros(obs_dim))

        # 雅可比计算缓存
        self.jacobian = None

    def compute_jacobian(self, f, x):
        """自动微分计算雅可比矩阵（EKF核心）"""
        with torch.enable_grad():
            B, D = x.shape
            x = x.detach().requires_grad_(True)
            y = f(x)
            jac = torch.zeros(B, D, D, device=x.device)
            for i in range(D):
                grad = torch.autograd.grad(
                    y[:, i].sum(), x, retain_graph=True, create_graph=True
                )[0]
                jac[:, i, :] = grad
        return jac

    def forward(self, obs, prev_state, prev_cov, t):
        B = prev_state.size(0)  # 获取batch大小

        # 过程噪声协方差（对角）
        Q = torch.diag_embed(torch.exp(self.log_Q_diag)).unsqueeze(0).repeat(B, 1, 1)

        # 观测噪声协方差
        R = torch.diag_embed(torch.exp(self.log_R_diag)).unsqueeze(0).repeat(B, 1, 1)

        # 状态转移函数及其雅可比
        def f(x):
            return self.state_transfer(x, t)

        self.jacobian = self.compute_jacobian(f, prev_state)
        F = self.jacobian

        # === 预测步骤 ===
        state_pred = f(prev_state)
        cov_pred = torch.bmm(torch.bmm(F, prev_cov), F.transpose(1, 2)) + Q

        # === 更新步骤 ===
        # 观测预测：z_pred = H * state_pred
        obs_pred = state_pred @ self.obs_matrix.t()  # (B, obs_dim)
        residual = (obs - obs_pred).unsqueeze(-1)  # (B, obs_dim, 1)

        # 卡尔曼增益
        H = self.obs_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, obs_dim, state_dim)
        S = torch.bmm(H, torch.bmm(cov_pred, H.transpose(1, 2))) + R
        S_inv = torch.linalg.pinv(S)  # 伪逆保证数值稳定
        K = torch.bmm(torch.bmm(cov_pred, H.transpose(1, 2)), S_inv)  # (B, state_dim, obs_dim)

        # 状态更新
        state_update = state_pred + torch.bmm(K, residual).squeeze(-1)

        # 协方差更新（Joseph形式）
        I = torch.eye(self.state_dim, device=obs.device).unsqueeze(0).expand(B, -1, -1)
        I_KH = I - torch.bmm(K, H)
        cov_update = torch.bmm(I_KH, torch.bmm(cov_pred, I_KH.transpose(1, 2)))
        cov_update += torch.bmm(torch.bmm(K, R), K.transpose(1, 2))

        return state_update, cov_update, torch.diagonal(cov_update, dim1=1, dim2=2)


class MultiScaleKalmanFilter(nn.Module):
    """多尺度卡尔曼-切比雪夫网络（创新点3）"""

    def __init__(self, input_dim, state_dim, seq_len, base_degree=4, num_scales=2):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.num_scales = num_scales

        # 多尺度处理层
        self.scales = nn.ModuleList()
        for i in range(num_scales):
            factor = 2 ** i
            scaled_len = max(1, seq_len // factor)  # 确保至少为1
            self.scales.append(
                DifferentiableKalmanCell(
                    state_dim, input_dim, base_degree, scaled_len
                )
            )

        # 跨尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(state_dim * num_scales * 2, 256),  # 包含状态和不确定性
            nn.GELU(),
            nn.Linear(256, state_dim)
        )

        # 不确定性量化输出
        self.uncertainty_proj = nn.Linear(state_dim, 2)

    def forward(self, x):
        """输入: (batch, seq_len, input_dim)"""
        B, T, D = x.shape
        device = x.device

        # 初始化状态
        states = torch.zeros(B, self.state_dim, device=device)
        covs = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(B, 1, 1)

        # 多尺度处理
        scale_outputs = []
        for i, cell in enumerate(self.scales):
            scale_factor = 2 ** i
            # 下采样输入
            if scale_factor > 1:
                scaled_input = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=scale_factor, stride=scale_factor)
            else:
                scaled_input = x.permute(0, 2, 1)

            scaled_input = scaled_input.permute(0, 2, 1)
            scale_len = scaled_input.size(1)

            scale_states = []
            scale_uncertainties = []
            current_states = states.clone()
            current_covs = covs.clone()

            for t in range(scale_len):
                current_states, current_covs, diag_cov = cell(
                    scaled_input[:, t], current_states, current_covs, t
                )
                scale_states.append(current_states.unsqueeze(1))
                scale_uncertainties.append(diag_cov.unsqueeze(1))

            # 上采样到原始序列长度
            states_seq = torch.cat(scale_states, dim=1)
            states_seq = F.interpolate(
                states_seq.permute(0, 2, 1),
                size=T,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)

            unc_seq = torch.cat(scale_uncertainties, dim=1)
            unc_seq = F.interpolate(
                unc_seq.permute(0, 2, 1),
                size=T,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)

            # 合并状态和不确定性
            scale_output = torch.cat([states_seq, unc_seq], dim=-1)
            scale_outputs.append(scale_output)

        # 跨尺度融合
        all_states = torch.cat(scale_outputs, dim=-1)  # (B, T, state_dim * num_scales * 2)
        fused_state = self.fusion(all_states)

        # 不确定性量化（创新点4）
        uncertainty = self.uncertainty_proj(fused_state)
        mean, log_var = torch.chunk(uncertainty, 2, dim=-1)
        std = torch.exp(0.5 * log_var)

        return {
            "state": fused_state,
            "mean": mean,
            "std": std,
            "scale_outputs": scale_outputs
        }


class ChebyshevKalmanNetwork(nn.Module):
    """端到端可训练的卡尔曼-切比雪夫网络"""

    def __init__(self, input_dim, output_dim, state_dim=64, seq_len=96,
                 base_degree=4, num_scales=3):
        super().__init__()
        # 编码器 - 对每个线性层单独应用谱归一化
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 128)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(128, state_dim))
        )
        self.kalman_filter = MultiScaleKalmanFilter(
            state_dim, state_dim, seq_len, base_degree, num_scales
        )

        # 解码器 - 对每个线性层单独应用谱归一化
        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(state_dim, 128)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(128, output_dim))
        )

        # 时间位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, state_dim))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

        # 序列长度
        self.seq_len = seq_len

    def forward(self, x):
        # 输入: (batch, seq_len, input_dim)
        if x.size(1) != self.seq_len:
            # 动态调整位置编码
            pe = F.interpolate(self.positional_encoding.permute(0, 2, 1),
                               size=x.size(1),
                               mode='linear',
                               align_corners=True).permute(0, 2, 1)
        else:
            pe = self.positional_encoding

        encoded = self.encoder(x)

        # 添加位置编码
        encoded = encoded + pe

        kalman_out = self.kalman_filter(encoded)
        decoded = self.decoder(kalman_out["state"])

        # 返回预测和不确定性
        return {
            "prediction": decoded,
            "mean": kalman_out["mean"],
            "std": kalman_out["std"]
        }


# 测试代码
if __name__ == "__main__":
    # 模拟工业级数据 (batch=32, seq=96, features=12)
    input_tensor = torch.randn(32, 96, 12)

    # 初始化模型
    model = ChebyshevKalmanNetwork(
        input_dim=12,
        output_dim=1,  # 预测单变量
        state_dim=64,
        seq_len=96,
        base_degree=4,
        num_scales=3
    )

    # 前向传播
    results = model(input_tensor)

    print("预测形状:", results["prediction"].shape)
    print("均值形状:", results["mean"].shape)
    print("标准差形状:", results["std"].shape)

    # 验证不确定性量化
    assert torch.all(results["std"] > 0), "标准差应为正值"
    print("测试通过！模型输出包含预测和不确定性量化。")