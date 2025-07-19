import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Base ChebyKANLinear from original implementation
# Now treats last dimension as feature axis (sequence length)
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1, dtype=torch.float32))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False

    def forward(self, x):
        # x: (B, V, T) or reshaped to (B*V, T)
        b_v, dim = x.shape
        assert dim == self.inputdim, f"Expected last dim {self.inputdim}, got {dim}"
        if self.pre_mul:
            mul_1 = x[:, ::2]
            mul_2 = x[:, 1::2]
            x = torch.cat([x[:, :dim//2], mul_1 * mul_2], dim=-1)

        # Expand for Chebyshev basis
        x_exp = x.view(b_v, dim, 1).expand(-1, -1, self.degree + 1)
        t = torch.tanh(x_exp).clamp(-1 + self.epsilon, 1 - self.epsilon)
        theta = torch.acos(t)
        arange = self.arange
        cheb = torch.cos(theta * arange)
        cheb = cheb
        cheby_coeffs = self.cheby_coeffs
        y = torch.einsum("bid,iod->bo", cheb, cheby_coeffs)
        y = y.view(b_v, self.outdim)

        if self.post_mul:
            mul_1 = y[:, ::2]
            mul_2 = y[:, 1::2]
            y = torch.cat([y[:, :self.outdim//2], mul_1 * mul_2], dim=-1)
        return y



# 2. Kalman-Cheby Filter Layer (fixed dimension handling)
class KalmanChebyLayer(nn.Module):
    def __init__(self, seqlen, degree):
        super().__init__()
        self.predictor = ChebyKANLinear(seqlen, seqlen, degree)
        self.measure_map = nn.Linear(seqlen, seqlen)
        self.R = nn.Parameter(torch.eye(seqlen))
        self.Q = nn.Parameter(torch.eye(seqlen))

    def forward(self, meas, state, P):
        # meas, state: (B, T); P: (B, T, T)
        B, T = meas.shape
        x_pred = self.predictor(state)
        P_pred = P + self.Q.unsqueeze(0)
        z = self.measure_map(meas)
        z_pred = self.measure_map(x_pred)
        residual = (z - z_pred).unsqueeze(-1)       # (B,T,1)
        H = self.measure_map.weight               # (T,T)
        S = H @ P_pred @ H.t() + self.R.unsqueeze(0) # (B,T,T)
        S_inv = torch.linalg.inv(S)
        K = P_pred @ H.t() @ S_inv                # (B,T,T)
        state_upd = x_pred.unsqueeze(-1) + K @ residual
        state_upd = state_upd.squeeze(-1)
        P_upd = (torch.eye(T, device=meas.device) - K @ H) @ P_pred
        state_ref = self.predictor(state_upd)
        return state_ref, P_upd

class BatchKalmanCheby(nn.Module):
    def __init__(self, seqlen, degree):
        super().__init__()
        self.layer = KalmanChebyLayer(seqlen, degree)

    def forward(self, x):
        # x, states: (B, V, T); Ps: (B, V, T, T)
        B, V, T = x.shape
        states = torch.zeros(B, V, T).to(x.device)
        Ps = torch.eye(T).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1).to(x.device)

        B, V, T = x.shape
        meas = x.reshape(B*V, T)
        state = states.reshape(B*V, T)
        P = Ps.reshape(B*V, T, T)

        state_ref,_ = self.layer(meas, state, P)
        state_ref = state_ref.view(B, V, T)
        # P_upd = P_upd.view(B, V, T, T)
        return state_ref#, P_upd


# ----------------------
# Testing Main Function
# ----------------------
def main():
    B, V, T = 32, 7, 96
    x = torch.rand(B, V, T)


    # 2. KalmanChebyLayer
    bkc = BatchKalmanCheby(T, degree=3)
    states = torch.zeros(B, V, T)
    Ps = torch.eye(T).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    out_state, out_P = bkc(x, states, Ps)
    print('BatchKalmanCheby state:', out_state.shape, 'cov:', out_P.shape)

if __name__ == "__main__":
    main()
