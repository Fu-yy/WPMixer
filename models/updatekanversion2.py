import torch
import torch.nn as nn


class ChebyKANLinear(nn.Module):
    """
    Chebyshev 多项式映射：
      输入 x: (Bv, T)  → 输出 y: (Bv, T)
    """
    def __init__(self, dim, degree):
        super().__init__()
        self.dim = dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(dim, dim, degree+1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0/(dim*(degree+1)))

        self.register_buffer("arange", torch.arange(degree+1, dtype=torch.float32))
        self.eps = 1e-7

    def forward(self, x):
        # x: (Bv, T)
        Bv, T = x.shape
        assert T == self.dim

        # 构造 Chebyshev 基
        x3 = x.unsqueeze(-1).expand(-1, -1, self.degree+1)  # (Bv, T, d+1)
        t = torch.tanh(x3).clamp(-1+self.eps, 1-self.eps)
        theta = torch.acos(t)                                 # (Bv, T, d+1)
        cheb = torch.cos(theta * self.arange)                # (Bv, T, d+1)

        # einsum 得到 (Bv, T)
        y = torch.einsum("bik,ijk->bj", cheb, self.cheby_coeffs)
        return y


class KalmanChebyLayer(nn.Module):
    """
    单步 Kalman+Cheby 更新：
      meas, state: (Bv, T)
      P:          (Bv, T, T)
    返回更新后 state_upd:(Bv,T)  和  P_upd:(Bv,T,T)
    """
    def __init__(self, seqlen, degree):
        super().__init__()
        self.seqlen = seqlen

        # 预测模型
        self.predictor = ChebyKANLinear(seqlen, degree)
        # 观测映射 H
        self.measure_map = nn.Linear(seqlen, seqlen, bias=False)

        # 用于构造正定 Q, R
        self.Lq = nn.Parameter(torch.eye(seqlen))
        self.Lr = nn.Parameter(torch.eye(seqlen))
        self.eps = 1e-6

    def get_Q(self, Bv):
        L = torch.tril(self.Lq)
        Q = L @ L.t() + self.eps * torch.eye(self.seqlen, device=L.device)
        return Q.unsqueeze(0).expand(Bv, self.seqlen, self.seqlen)

    def get_R(self, Bv):
        L = torch.tril(self.Lr)
        R = L @ L.t() + self.eps * torch.eye(self.seqlen, device=L.device)
        return R.unsqueeze(0).expand(Bv, self.seqlen, self.seqlen)

    def forward(self, meas, state, P):
        # meas, state: (Bv, T);  P: (Bv, T, T)
        Bv, T = state.shape

        # -- 1) 预测
        x_pred = self.predictor(state)              # (Bv, T)
        P_pred = P + self.get_Q(Bv)                 # (Bv, T, T)

        # -- 2) 更新
        z      = self.measure_map(meas)             # (Bv, T)
        z_pred = self.measure_map(x_pred)           # (Bv, T)

        H = self.measure_map.weight                  # (T, T)
        R = self.get_R(Bv)                           # (Bv, T, T)

        S     = H @ P_pred @ H.t() + R               # (Bv, T, T)
        S_inv = torch.linalg.inv(S)                  # (Bv, T, T)
        K     = P_pred @ H.t().unsqueeze(0).expand(Bv,T,T) @ S_inv  # (Bv,T,T)

        residual = (z - z_pred).unsqueeze(-1)        # (Bv, T, 1)
        state_upd = x_pred.unsqueeze(-1) + K @ residual  # (Bv, T, 1)
        state_upd = state_upd.squeeze(-1)            # (Bv, T)

        I = torch.eye(T, device=state.device).unsqueeze(0).expand(Bv, T, T)
        P_upd = (I - K @ H.unsqueeze(0)) @ P_pred     # (Bv, T, T)

        return state_upd, P_upd


class BatchKalmanCheby(nn.Module):
    """
    批量单步滤波：
      输入 x:(B,V,T)
      输出 state_ref:(B,V,T), cov:(B,V,T,T)
    """
    def __init__(self, seqlen, degree):
        super().__init__()
        self.layer = KalmanChebyLayer(seqlen, degree)

    def forward(self, x):
        B, V, T = x.shape
        Bv = B * V

        # 将每一条 (batch,channel) 看作一个独立序列
        meas  = x.reshape(Bv, T)               # (Bv, T)
        state = torch.zeros(Bv, T, device=x.device)
        P     = torch.eye(T, device=x.device).unsqueeze(0).expand(Bv, T, T)

        state_ref, P_upd = self.layer(meas, state, P)

        # 恢复回 (B,V,...)
        state_ref = state_ref.view(B, V, T)
        cov       = P_upd.view(B, V, T, T)

        return state_ref, cov


# ---------------
# 测试
# ---------------
if __name__ == "__main__":
    B, V, T = 4, 3, 10
    x = torch.randn(B, V, T)
    model = BatchKalmanCheby(T, degree=3)
    states, covs = model(x)
    print("输出状态 shape:", states.shape)   # (4,3,10)
    print("输出协方差 shape:", covs.shape)    # (4,3,10,10)
