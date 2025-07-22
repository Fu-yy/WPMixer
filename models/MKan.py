import torch
import torch.nn as nn
import torch.nn.functional as F


"""《TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting》 ICLR 2025
现实世界的时间序列通常具有多个相互交织的频率成分，这使得准确的时间序列预测具有挑战性。将混合频率成分分解为多个单频率成分是一种自然的选择。
然而，模式的信息密度在不同频率上有所不同，对不同频率成分采用统一的建模方法可能会导致不准确的表征。
为了应对这一挑战，受最近的 Kolmogorov-Arnold 网络 (KAN) 灵活性的启发，我们提出了一种基于 KAN 的频率分解学习架构 (TimeKAN)，以解决由多频率混合引起的复杂预测挑战。
具体来说，TimeKAN 主要由三个组件组成：级联频率分解 (CFD) 块、多阶 KAN 表示学习 (M-KAN) 块和频率混合块。CFD 块采用自下而上的级联方法获得每个频带的序列表示。
得益于 KAN 的高灵活性，我们设计了一个新颖的 M-KAN 块来学习和表示每个频带内的特定时间模式。最后，使用频率混合块将频带重新组合为原始格式。
在多个真实世界时间序列数据集上进行的大量实验结果表明，TimeKAN 作为一种极轻量级架构实现了最先进的性能。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.kaiming_normal_(self.cheby_coeffs, mode='fan_in', nonlinearity='linear')
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        # x = torch.tanh(x)
        x = torch.tanh(x)
        # x = torch.acos(x)
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        arange = self.arange.to(x.device)
        x = x* arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        cheby_coeffs = self.cheby_coeffs.to(x.device)
        y = torch.einsum(
            "bid,iod->bo", x, cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        return y



class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.in_features = in_features
        self.fc1 = ChebyKANLinear(
            in_features,
            out_features,
            order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x



class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree, stride=1, padding=0, dilation=1, groups=1, act=False, bn=False,
                 bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class M_KAN(nn.Module):
    def __init__(self, input_dim,output_dim, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(input_dim, output_dim, order)
        )
        self.conv = BasicConv(input_dim, output_dim, kernel_size=3, degree=order, groups=input_dim)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1+x2
        return out


# 完全替换原有的 ChebyKANLinear 类
class EfficientChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, groups=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.groups = groups

        # 分组参数化提高并行性
        self.weight = nn.Parameter(torch.empty(groups, degree + 1, input_dim // groups, output_dim))
        self.scale = nn.Parameter(torch.ones(output_dim))
        nn.init.normal_(self.weight, 0, 1 / (input_dim * (degree + 1)))

        # 预计算多项式基函数
        self.register_buffer('cheby_coeffs', self._precompute_coeffs(degree), persistent=False)

    def _precompute_coeffs(self, degree):
        # 利用递推关系预计算系数矩阵
        coeffs = torch.zeros(degree + 1, degree + 1)
        coeffs[0, 0] = 1
        if degree >= 1:
            coeffs[1, 1] = 1
            for k in range(2, degree + 1):
                coeffs[k] = 2 * F.pad(coeffs[k - 1], (1, 0))[:degree + 1] - coeffs[k - 2]
        return coeffs

    def forward(self, x):
        # 分组处理提升效率
        x = x.view(-1, self.groups, self.input_dim // self.groups)

        # 多项式快速计算 (避免数值不稳定的acos)
        x_norm = torch.tanh(x)  # 映射到[-1,1]
        poly_vals = torch.zeros(x.shape[0], self.groups, self.degree + 1, device=x.device)

        # 使用递推关系计算多项式值
        poly_vals[..., 0] = 1
        if self.degree >= 1:
            poly_vals[..., 1] = x_norm
            for k in range(2, self.degree + 1):
                poly_vals[..., k] = 2 * x_norm * poly_vals[..., k - 1] - poly_vals[..., k - 2]

        # 线性组合
        weighted = torch.einsum('bgk,gkdo->bgdo', poly_vals, self.weight)
        y = weighted.sum(dim=1)  # [batch, groups, output_dim] -> [batch, output_dim]
        return y * self.scale


# 修改 ChebyKANLayer 以使用新实现
class EfficientChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, order, groups=4):
        super().__init__()
        self.fc = EfficientChebyKANLinear(input_dim, output_dim, order, groups)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        x = self.fc(x)
        return x.view(*orig_shape[:-1], -1)

if __name__ == '__main__':
    batch_size = 16
    seq_len = 7
    d_model = 12
    order = 3
    new  = EfficientChebyKANLayer(input_dim=d_model,output_dim=d_model, order=order)
    block = M_KAN(input_dim=d_model,output_dim=d_model, order=order).to('cuda')

    input = torch.rand(batch_size, seq_len, d_model).to('cuda')
    new_out =new(input)
    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")


