import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from layers.revin import RevIN

# 可学习频率门控的动态频谱分解
class FrequencyGuidedDecomposition(nn.Module):
    def __init__(self, seq_len, n_var, top_k=10):
        super(FrequencyGuidedDecomposition, self).__init__()
        self.seq_len = seq_len
        self.n_var = n_var
        self.top_k = top_k
        self.freq_weights = nn.Parameter(torch.randn(1, n_var, seq_len // 2 + 1))

    def forward(self, x):
        # x: (batch, seq_len, n_var) -> (batch, n_var, seq_len)
        x_perm = x.permute(0, 2, 1)
        xf = torch.fft.rfft(x_perm, dim=-1)
        weight = torch.sigmoid(self.freq_weights)

        # Top-k频率选择
        topk_weight, indices = torch.topk(weight, self.top_k, dim=-1)
        mask = torch.zeros_like(weight).scatter_(-1, indices, topk_weight)

        xf_filtered = xf * mask
        x_filtered = torch.fft.irfft(xf_filtered, n=self.seq_len, dim=-1)

        # 转回原始维度顺序
        x_filtered = x_filtered.permute(0, 2, 1)
        nonstationary_component = x - x_filtered
        return x_filtered, nonstationary_component

def divide_into_segments(distance, divide_len):
    segment_size = distance / divide_len
    segments = []
    current_value = 0
    for _ in range(divide_len):
        segment = current_value
        segments.append(segment)
        current_value += segment_size
    return segments
def fourier_zero( x_enc,down_sampling_layers,d_model):



        '''
        :param x_enc:使用傅里叶   三分之一取点


        '''
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_sampling_list = []
        # x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        sample_rate = 4096*1

        freq, _ = torch.sort(abs(torch.fft.rfftfreq(d_model, 1 / sample_rate)))
        min_freq = freq[0]
        max_freq = freq[-1]
        distance = abs(max_freq - min_freq).item()
        divide_len_list = divide_into_segments(distance,divide_len = down_sampling_layers)

        fft_signal = torch.fft.rfft(x_enc, dim=2, norm='ortho')  # signal_data为长度8192的list

        for index in range(len(divide_len_list)):
            cut_fft = fft_signal.clone()
            # mask = torch.zeros_like(cut_fft, device=fft_signal.device)
            mask = torch.ones_like(cut_fft, dtype=torch.bool, device=fft_signal.device)  # 创建一个与cut_fft同样大小的布尔mask
            #  2025-02-10 23:06:48 新增 begin
            # 设置低于当前分割点的频率位置为0
            mask[:, :, freq < divide_len_list[index]] = 0
            # mask = (freq > divide_len_list[index + 1])  # 创建一个布尔mask
            # cut_fft = cut_fft * ~mask.unsqueeze(0).unsqueeze(0).to(cut_fft.device)  # 用mask将指定位置置为0
            # 如果不是最后一个区间，设置高于下一个分割点的频率为0
            if index + 1 < len(divide_len_list):
                mask[:, :, freq > divide_len_list[index + 1]] = 0
            # 使用mask将不需要的部分置为0
            cut_fft_mask = cut_fft * mask
            cut_fft_signal_mask = torch.fft.irfft(cut_fft_mask,n=x_enc.shape[-1],dim=2, norm='ortho')

            x_enc_sampling_list.append(cut_fft_signal_mask.real.permute(0, 2, 1).contiguous())

        return x_enc_sampling_list
def mvmd_decompose(x: torch.Tensor, K: int, alpha: float = 2000, tau: float = 0.0,
                   tol: float = 1e-7, max_iter: int = 500):
    """
    多变量变分模态分解 (MVMD)。
    参数:
        x: 输入信号张量，形状 (batch, seq_len, n_var)
        K: 希望提取的模态数量
        alpha: 模态带宽惩罚系数 (alpha越大则提取的模式带宽越窄，频率分离度越高)
        tau: 对偶梯度上升步长参数 (tau=0 则允许噪声残留, tau=1 则严格重构约束)
        tol: 收敛判据阈值 (相对变化量)
        max_iter: 最大迭代次数
    返回:
        imfs_list: 长度为K的列表，每个元素形状 (batch, seq_len, n_var), 对应各模态分解信号
    """
    device = x.device
    batch, seq_len, n_var = x.shape
    # 将输入转为float类型张量
    signal = x.to(torch.float32)
    # 在时间轴上对每个通道进行FFT（采用实FFT只保留非负频率部分）
    # 输出 shape: (batch, n_var, Nf), 其中 Nf = floor(seq_len/2) + 1
    X = torch.fft.rfft(signal.transpose(1, 2), dim=2)
    batch, n_var, Nf = X.shape
    # 生成对应的频率坐标 [0, 1/2] (归一化频率，单位: 周期/采样间隔)
    # 如果需要物理频率，可根据采样率Fs换算为 [0, Fs/2]
    freqs = torch.linspace(0, 0.5, Nf, device=device)
    # 初始化模态频谱 U_k 和中心频率 omega_k
    U = torch.zeros((batch, K, n_var, Nf), dtype=X.dtype, device=device)
    # 初始化各模态中心频率为均匀分布在频带内
    omega = torch.zeros((batch, K), device=device)
    # 我们选取初始频率索引均分频谱长度
    for k in range(K):
        idx = int((k + 1) * (Nf - 1) / (K + 1))
        omega[:, k] = freqs[idx]  # 每个batch采用相同初始值，可随机扰动不同batch以增加多样性
    # 初始化拉格朗日乘子 Lambda
    Lambda = torch.zeros((batch, n_var, Nf), dtype=X.dtype, device=device)
    # ADMM 迭代
    prev_U = None
    for n in range(max_iter):
        # 保存前一步结果用于监测收敛
        prev_U = U.clone()
        # **1. 模态更新 (频域并行向量化计算)**
        # 计算当前所有模态谱之和 S = sum_k U_k  (shape: (batch, n_var, Nf))
        S = U.sum(dim=1)
        # 为广播，将 X, S, Lambda 扩展出模态维度
        X_exp = X.unsqueeze(1)  # (batch, 1, n_var, Nf)
        S_exp = S.unsqueeze(1)  # (batch, 1, n_var, Nf)
        Lambda_exp = Lambda.unsqueeze(1)  # (batch, 1, n_var, Nf)
        # 计算每个模态的新谱
        # 分母: 1 + 2*alpha*(omega - omega_k)^2
        # 将omega_k (batch, K)与频率网格计算差平方
        # 扩展omega_k为 (batch, K, 1, 1) 以便与freqs (1,1,1,Nf)做广播
        omega_exp = omega[:, :, None, None]  # (batch, K, 1, 1)
        freq_grid = freqs[None, None, None, :]  # (1, 1, 1, Nf)
        denom = 1 + 2 * alpha * (freq_grid - omega_exp) ** 2  # (batch, K, 1, Nf)
        # 分子: X - (S - U) + (Lambda/2)
        # 由于当前U仍是上一步的值，我们直接利用其计算
        numerator = X_exp - (S_exp - U) + 0.5 * Lambda_exp  # (batch, K, n_var, Nf)
        # 更新U_k谱
        U = numerator / denom  # 广播 denom 到每个通道

        # **2. 更新中心频率 omega_k**
        # 计算每个模态的功率谱 (将各通道功率相加)
        # U.shape: (batch, K, n_var, Nf)
        power = (U.real ** 2 + U.imag ** 2).sum(dim=2)  # (batch, K, Nf)
        # 频率数组freqs扩展为 (1,1,Nf) 方便批和模态维度计算加权平均
        freq_grid2 = freqs[None, None, :]
        # 计算加权平均频率
        omega_num = (power * freq_grid2).sum(dim=2)  # (batch, K)
        omega_den = power.sum(dim=2) + 1e-12  # 避免除零
        omega = omega_num / omega_den  # 更新中心频率 (batch, K)

        # **3. 拉格朗日乘子更新**
        # 更新Lambda以减少重构误差: Lambda = Lambda + tau * (X - S_new)
        S_new = U.sum(dim=1)  # (batch, n_var, Nf)
        # 广播 tau 为实数，与复数 Lambda 匹配计算
        Lambda = Lambda + tau * (X - S_new)

        # **4. 检查收敛条件**
        # 计算所有模态谱的变化 (L2范数) 相对于数据范数的比值
        diff = (U - prev_U).norm() / (prev_U.norm() + 1e-12)
        if diff.item() < tol:
            break
    # ADMM迭代结束，将频域模态谱转换回时域信号
    # 对每个模态做逆FFT (利用irfft从实频谱重建实信号)
    # U当前shape: (batch, K, n_var, Nf)
    # 直接对最后一维执行irfft，指定输出长度为原序列长度
    u_time = torch.fft.irfft(U, n=seq_len, dim=-1)  # 输出shape: (batch, K, n_var, seq_len)
    # 调整维度为 (batch, seq_len, n_var, K) 再拆分
    u_time = u_time.permute(0, 3, 2, 1)  # (batch, seq_len, n_var, K)
    imfs_list = list(u_time.unbind(dim=-1))  # 长度K，每个shape (batch, seq_len, n_var)
    return imfs_list

def mvmd_decompose_new(x: torch.Tensor, K: int, alpha: float = 2000, tau: float = 0.0,
                   tol: float = 1e-7, max_iter: int = 100, ema_alpha: float = 0.9,rate: float = 0.3, drop_last: bool = False):
    """
    优化版多变量变分模态分解 (MVMD)：
        - 采用并行计算 (方法1)
        - 预计算闭式解减少计算量 (方法2)
        - 自适应终止策略 (方法3)

    参数:
        x: 输入信号张量，形状 (batch, seq_len, n_var)
        K: 希望提取的模态数量
        alpha: 模态带宽惩罚系数
        tau: 对偶梯度上升步长参数
        tol: 收敛判据阈值 (相对变化量)
        max_iter: 最大迭代次数
        ema_alpha: 指数移动平均参数 (用于自适应终止)
        rate: 前百分之多少个模态相加
        drop_last： 是否丢弃最后一个模态

    返回:
        imfs_list: 长度为K的列表，每个元素形状 (batch, seq_len, n_var)
    """
    device = x.device
    batch, seq_len, n_var = x.shape

    # 转换为 float 并计算 FFT
    signal = x.to(torch.float32)
    X = torch.fft.rfft(signal.transpose(1, 2), dim=2)  # (batch, n_var, Nf)
    batch, n_var, Nf = X.shape

    # 生成频率坐标
    freqs = torch.linspace(0, 0.5, Nf, device=device)

    # 初始化模态频谱 U_k, 中心频率 omega_k
    U = torch.zeros((batch, K, n_var, Nf), dtype=X.dtype, device=device)

    # 采用均匀初始化方式选择中心频率
    omega = freqs[(torch.linspace(1, Nf-1, K, device=device).long())].expand(batch, K)

    # 初始化拉格朗日乘子 Lambda
    Lambda = torch.zeros((batch, n_var, Nf), dtype=X.dtype, device=device)

    # **改进: 自适应终止策略**
    prev_diff = float('inf')
    ema_diff = float('inf')  # 指数移动平均 (EMA) 用于平滑终止判据

    for n in range(max_iter):
        prev_U = U.clone()

        # **方法 1: 并行计算所有模态**
        S = U.sum(dim=1, keepdim=True)  # (batch, 1, n_var, Nf)
        X_exp = X.unsqueeze(1)  # (batch, 1, n_var, Nf)
        Lambda_exp = Lambda.unsqueeze(1)  # (batch, 1, n_var, Nf)

        # **方法 2: 预计算闭式解**
        omega_exp = omega[:, :, None, None]  # (batch, K, 1, 1)
        freq_grid = freqs[None, None, None, :]  # (1, 1, 1, Nf)
        denom = 1 + 2 * alpha * (freq_grid - omega_exp) ** 2  # 计算所有模态的分母

        # 计算 U_k 并行更新 (优化分子计算)
        numerator = X_exp - S + U + 0.5 * Lambda_exp
        U = numerator / denom  # 并行更新所有模态

        # **方法 2: 预计算闭式解用于更新 omega_k**
        # 计算每个模态的功率谱 (将各通道功率相加)
        power = (U.real ** 2 + U.imag ** 2).sum(dim=2)  # (batch, K, Nf)

        # 确保频率数组 freq_grid 形状正确 (batch, K, Nf)
        freq_grid = freqs.view(1, 1, -1).to(device)  # (1, 1, Nf)

        # 计算加权平均频率
        omega_num = (power * freq_grid).sum(dim=2)  # (batch, K)
        omega_den = power.sum(dim=2) + 1e-12  # 避免除零
        omega = omega_num / omega_den  # (batch, K) 更新中心频率

        # **方法 1: 计算 Lambda 并行更新**
        S_new = U.sum(dim=1)
        Lambda = Lambda + tau * (X - S_new)

        # **方法 3: 自适应终止策略**
        diff = (U - prev_U).norm() / (prev_U.norm() + 1e-12)
        ema_diff = ema_alpha * ema_diff + (1 - ema_alpha) * diff.item()  # 指数平滑

        if ema_diff < tol:
            break  # 自适应终止

    # 逆FFT转换回时域
    u_time = torch.fft.irfft(U, n=seq_len, dim=-1)  # (batch, K, n_var, seq_len)

    # 调整维度并返回分解结果
    u_time = u_time.permute(0, 3, 2, 1)  # (batch, seq_len, n_var, K)
    imfs_list = list(u_time.unbind(dim=-1))  # 转换为列表
    res_list = []

    processed = imfs_list[:-1] if drop_last else imfs_list.copy()
    num_tensors = len(processed)
    take_num = int(num_tensors * rate)
    take_num = max(0, min(take_num, num_tensors))  # 确保取值在合理范围

    # 获取目标张量子集
    selected_front = processed[:take_num]
    selected_rear = processed[take_num:]
    res_front = torch.stack(selected_front).sum(0)

    res_rear = torch.stack(selected_rear).sum(0)


    # 阶段3：
    return x, res_front,res_rear

class xPathModel(nn.Module):
    def __init__(self, configs,item):
        super(xPathModel, self).__init__()
        self.configs = configs
        # Parameters
        seq_len = item   # lookback window L
        pred_len = item # prediction length (96, 192, 336, 720)
        c_in = configs.c_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        patch_len = 8
        # stride_list = [2 ,7 ,6 ,2]
        stride = 8


        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)
        self.revin_layer_rear = RevIN(c_in,affine=True,subtract_last=False)
        self.revin_layer_front = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        # self.decomp = FrequencyGuidedDecomposition(seq_len=seq_len,n_var=c_in,top_k=10)

        self.net_list = nn.ModuleList()
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)

        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream

    def forward(self, x):
        # x: [Batch, Input, Channel]
        x = x.permute(0,2,1)
        # Normalization
        # if self.revin:
        #     x = self.revin_layer(x, 'norm')
        #     # res_front = self.revin_layer_front(res_front, 'norm')
        #     # res_rear = self.revin_layer_rear(res_rear, 'norm')

        # --------------------------------------------- begin

        # res_list = []
        # if self.configs.use_fourier == 1:
        #     res_list = fourier_zero(x,self.configs.down_sample_layers,self.configs.seq_len)
        # else:
        #     for _ in range(self.configs.down_sample_layers):
        #         res_list.append(x)
        # --------------------------------------------- end

        # res_list = mvmd_decompose_new(x,6)

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            # trend_init_1,seasonal_init_1 = fourier_zero(x, 2, 96)
            x_res_list = []
            # --------------------------------------------- begin
            #
            # for i,(imf,net) in enumerate(zip(res_list, self.net)):
            #     seasonal_init_1, trend_init_1 = self.decomp(imf)
            #     res_ts = net(seasonal_init_1, trend_init_1)
            #     x_res_list.append(res_ts)
            # --------------------------------------------- end

            seasonal_init, trend_init  = self.decomp(x)

            # x, res_front, res_rear = mvmd_decompose_new(x, K=6, rate=0.3, drop_last=False)

            x = self.net(seasonal_init, trend_init)
            # IMF1,IMF2 = x,x
            # seasonal_init_1, trend_init_1 = self.decomp(IMF1)
            # x_1 = self.net[0](seasonal_init_1, trend_init_1)
            # seasonal_init_2, trend_init_2 = self.decomp(IMF2)
            # x_2 = self.net[1](seasonal_init_2, trend_init_2)
            # x = x_1 + x_2
        # --------------------------------------------- begin
        # x = torch.stack(x_res_list,dim=-1).sum(-1)
        # --------------------------------------------- end

        # Denormalization
        # if self.revin:
        #     x = self.revin_layer(x, 'denorm')
        #     # x_front = self.revin_layer_front(x_front, 'denorm')
        #     # x_rear = self.revin_layer_rear(x_rear, 'denorm')

        return x.permute(0,2,1)