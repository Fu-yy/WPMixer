import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from models.MKan import M_KAN, ChebyKANLayer
from models.updatekanversion2 import  BatchKalmanCheby


# ------------------------
# 公共组件：升降块 & 逆升降方案
# ------------------------
class LiftingBlock(nn.Module):
    """
    Predict-Update 单级升降模块，仅用于逆变换。
    """
    def __init__(self, channels, kernel_size=4,d_model=12,order=3,device='cpu'):
        super().__init__()
        pad = (kernel_size // 2, kernel_size - 1 - kernel_size // 2)
        # self.P = nn.Sequential(
        #     nn.ReflectionPad1d(pad),
        #     nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     nn.GELU()
        # )
        # self.P = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU()
        # )
        #
        # self.U = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU()
        # )
        # self.U = nn.Sequential(
        #     nn.ReflectionPad1d(pad),
        #     nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     nn.GELU()
        # )
##################################################
#
#
# WPMixer_ETTh1_dec-True_sl96_pl96_dm256_bt32_wvdb2_tf5_df8_ptl16_stl8_sd42
# mse:0.39123696088790894, mae:0.4039541184902191
# datetime:2025-07-02  10:40:36  Wednesday
##################################################

        # self.P = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyAttentionBlock(nvar=channels, seqlen=d_model, degree=3, num_heads=2),
        #     nn.GELU(),
        # )
        #
        # self.U = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyAttentionBlock(nvar=channels, seqlen=d_model, degree=3, num_heads=2),
        #     nn.GELU(),
        #
        # )
##################################################
        ##################################################
        #
        #
        ##################################################
        # WPMixer_ETTh1_dec - True_sl96_pl96_dm256_bt32_wvdb2_tf5_df8_ptl16_stl8_sd42
        # mse: 0.3807249069213867, mae: 0.3989276587963104
        # datetime: 2025 - 07 - 02
        # 10: 45:56
        # Wednesday

        ##################################################

        # self.P = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     BatchDynamicDegreeCheby(d_model, max_degree=3),
        #     nn.GELU(),
        # )
        #
        # self.U = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     BatchDynamicDegreeCheby(d_model, max_degree=3),
        #     nn.GELU(),
        #
        # )
        ##################################################

        self.P = nn.Sequential(
            # nn.ReflectionPad1d(pad),
            # nn.Conv1d(channels, channels, kernel_size, groups=channels),
            BatchKalmanCheby(d_model, degree=3),
            nn.GELU(),
        ).to(device)

        self.U = nn.Sequential(
            # nn.ReflectionPad1d(pad),
            # nn.Conv1d(channels, channels, kernel_size, groups=channels),
            BatchKalmanCheby(d_model, degree=3),
            nn.GELU(),

        ).to(device)

        # self.P = nn.ModuleList([
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     BatchKalmanCheby(d_model, degree=3)
        #     for _ in range(2)
        # ]).to(device)
        # self.U = nn.ModuleList([
        #         # nn.ReflectionPad1d(pad),
        #         # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #         BatchKalmanCheby(d_model, degree=3)
        #     for _ in range(2)
        # ]).to(device)
        # self.act = nn.GELU()
        # self.P = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU(),
        # )
        #
        # self.U = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU(),
        #
        # )
        # self.P = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU(),
        # )
        #
        # self.U = nn.Sequential(
        #     # nn.ReflectionPad1d(pad),
        #     # nn.Conv1d(channels, channels, kernel_size, groups=channels),
        #     ChebyKANLayer(d_model, d_model, order),
        #     nn.GELU(),
        #
        # )


    def forward(self, x_even, x_odd, inverse=True):
        """
        仅支持 inverse=True 时的逆升降重构。
        x_even, x_odd: [B, C, L]
        返回 重构信号 [B, C, 2L]
        """
        if not inverse:
            raise RuntimeError("LiftingBlock.forward only supports inverse=True. Use lifting_forward for正变换.")
        # inverse 逆升降
        detail = x_odd
        approx = x_even



        x_even_rec = approx - self.U(detail)
        # x_even_rec =self.p_kan_block(x_even_rec)
        x_odd_rec = detail + self.P(x_even_rec)
        # x_odd_rec = self.u_kan_block(x_odd_rec)
        B, C, L = x_even_rec.shape
        x = x_even_rec.new_zeros((B, C, 2 * L))
        x[..., ::2] = x_even_rec
        x[..., 1::2] = x_odd_rec
        return x

def lifting_forward(x, P_block, U_block):
    """
    正向升降分解：返回 (approx, detail)
    x: [B, C, L]
    """
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    d = x_odd - P_block(x_even)
    c = x_even + U_block(d)
    return c, d

class TemporalAttentionGate(nn.Module):
    def __init__(self, levels, d_model, num_heads=1):
        super().__init__()
        self.levels = levels
        self.attn = nn.MultiheadAttention(embed_dim=levels, num_heads=num_heads)

    def forward(self, energy):  # energy: [B, levels, 2]
        # 展开为 [T=levels, B, C=2]
        e = energy.permute(2, 0, 1).reshape(2, -1, self.levels)
        attn_output, _ = self.attn(e, e, e)
        # 取第一时间步作为全局门控
        g = torch.sigmoid(attn_output[0]).reshape(-1, self.levels)
        return g
# ------------------------
# 方案 2 合并：MultiLevelLifting + 正则 + 逆升降
# ------------------------
class UnifiedMultiLevel(nn.Module):
    def __init__(self, channels=1, levels=3, k_size=4,
                 lambda_d=0.01, lambda_c=0.01,input_length=None,pred_length=None,d_model=None,batch_size=32,order=3,device='cpu'):
        super().__init__()
        self.levels = levels
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c
        self.device = device
        # 正向的 P/U 模块
        self.P_blocks = nn.ModuleList([
            LiftingBlock(channels, k_size,d_model=input_length // 2 ** (i+1),order=order,device=device).P for i in range(levels)
        ])
        self.U_blocks = nn.ModuleList([
            LiftingBlock(channels, k_size,d_model=input_length // 2 ** (i+1),order=order,device=device).U for i in range(levels)
        ])
        # 逆向完整 PU
        self.PU = nn.ModuleList([
            LiftingBlock(channels, k_size,d_model=input_length // 2 ** (i+1),order=order,device=device) for i in range(levels)
        ])
        # 门控：输入维度 levels*2, 输出 levels*2
        self.gate = nn.Linear((levels+1) * 2, (levels+1) * 2)
        # self.gate = TemporalAttentionGate(levels + 1, 2)

        # self.gate = nn.Sequential(
        #     nn.Linear(levels * 2, d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(d_model, levels * 2),
        #     nn.Sigmoid()  # 直接输出 [0,1] 门控值
        # )


        # self.coeff_kan_low = ChebyKANLayer(self.coeff_kan[-1].in_features, self.coeff_kan[-1].in_features, order)
        # self.coeff_kan.append(self.coeff_kan_low)

        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length, batch_size, channels,
                                               self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length, batch_size, channels,
                                              self.device)  # required length of the pred seq after decom

    def decompose(self, x):
        B, C, L = x.shape
        coeffs = []
        losses = torch.tensor(0.0, device=x.device)
        energy = []
        current = x
        sparse_losses = torch.tensor(0.0, device=x.device)

        # 正向升降 - 分解过程
        for i in range(self.levels):
            c, d = lifting_forward(current, self.P_blocks[i], self.U_blocks[i])
            coeffs.append(d)

            # 通道+时序平均，得到 [B,2]
            c_energy = c.abs().mean(dim=[1, 2])  # [B]
            d_energy = d.abs().mean(dim=[1, 2])  # [B]
            energy.append(torch.stack([c_energy, d_energy], dim=1))  # [B,2]

            # 正则项计算
            sparse_losses = sparse_losses + self.lambda_d * torch.mean(torch.abs(d))
            er = torch.norm(c) / (torch.norm(current[..., ::2]) + 1e-6) - 1
            sparse_losses = sparse_losses + self.lambda_c * (er ** 2)

            # losses = losses + self.lambda_d * F.smooth_l1_loss(d, torch.zeros_like(d))
            # er = torch.norm(c) / (torch.norm(current[..., ::2]) + 1e-6) - 1
            # losses = losses + self.lambda_c * (er ** 2)

            current = c

        coeffs.append(current)  # 最终近似系数
        # coeffs_new = []
        # for c,layer in zip(coeffs,self.coeff_kan):
        #     coeffs_item = layer(c)
        #     coeffs_new.append(coeffs_item)

        lo_c_energy = current.abs().mean(dim=[1, 2])  # [B]
        lo_d_energy = torch.zeros_like(lo_c_energy)  # [B]，因为 detail 不存在，就填 0

        energy.append(torch.stack([lo_c_energy, lo_d_energy], dim=1))  # [B,2]
        # lo_energy = current.abs().mean(dim=[1,2])  # [B]

        # energy = torch.stack(energy, dim=1)

        # 返回分解结果
        return coeffs, energy, sparse_losses
        # return {
        #     'coeffs': coeffs,  # 未应用门控的系数列表
        #     'energy': energy,  # 能量列表
        #     'losses': losses  # 正则项损失
        # }

    def reconstruct(self, coeffs, energy):
        # 从分解结果中提取数据
        # coeffs = decomposition_dict['coeffs']
        # energy = decomposition_dict['energy']

        B = coeffs[0].shape[0]  # 获取batch size

        # 计算门控
        e = energy  # [B, levels*2]
        e = torch.cat(energy, dim=1)  # [B, levels*2]

        g = torch.sigmoid(self.gate(e))  # [B, levels*2]
        # g = self.gate(e)  # [B, levels*2]
        # g = self.gate(e)  # [B, levels*2]

        # 应用门控到各系数 - 完全保留原始逻辑
        gated_coeffs = [coeffs[i] * g[:, i].view(-1, 1, 1) for i in range(len(coeffs))]

        # 逆升降重构
        rec = gated_coeffs[-1]
        for i in reversed(range(self.levels)):
            rec = self.PU[i](rec, gated_coeffs[i], inverse=True)
            # res = self.PU[i](rec, gated_coeffs[i], inverse=True)
            # rec = res + gated_coeffs[i] * 0.1  # 0.1 是残差缩放系数，可调

        return gated_coeffs, rec

    # 保持原有forward接口
    def forward(self, x):
        # 执行分解
        coeffs, energy, losses = self.decompose(x)

        # 执行重构
        gated_coeffs, rec = self.reconstruct(coeffs, energy)

        # 返回与原始格式相同的结果
        return gated_coeffs, rec, coeffs, energy, losses
    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        coeffs, energy, losses= self.decompose(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]
        yl = coeffs[-1]
        yh = coeffs[:-1]
        l = []
        l.append(yl.shape[-1])
        for c in yh:
            l.append(c.shape[-1])
        return l


# ------------------------
# 方案 3 合并：Fusion + 组件正则 + 逆升降
# ------------------------

class UnifiedFusion(nn.Module):
    def __init__(self, init_wavelet='db4', num_filters=8, levels=3,
                 channels=1, lambda_d=0.01, lambda_c=0.01, k_size=4,input_length=None,pred_length=None,batch_size=32,d_model=None,device='cpu'):
        super().__init__()

        self.device = device

        wavelet = pywt.Wavelet(init_wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32)
        K = dec_lo.numel()
        self.lo = nn.Parameter(dec_lo.view(1,1,K).repeat(num_filters,1,1))
        self.hi = nn.Parameter(dec_hi.view(1,1,K).repeat(num_filters,1,1))
        self.filter_w = nn.Parameter(torch.ones(num_filters))
        # asymmetric padding for conv1d
        self.pad = (K // 2, K - 1 - K // 2)
        self.stride = 2

        self.PU = nn.ModuleList([LiftingBlock(channels, k_size) for _ in range(levels)])
        self.levels = levels
        self.lambda_d, self.lambda_c = lambda_d, lambda_c

        self.norm_c = nn.GroupNorm(1, channels)
        self.norm_d = nn.GroupNorm(1, channels)

        dim_in = channels * levels * 2
        dim_out = levels * 2
        self.gate = nn.Linear(dim_in, dim_out)

        # 如果传入了 input_length，就预计算、存储输出尺寸
        self.input_w_dim = self._dummy_forward(input_length, batch_size, channels,
                                               self.device)  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(pred_length, batch_size, channels,
                                              self.device)  # required length of the pred seq after decom


    def decompose(self, x):
        B, C, L = x.shape
        dets, en_feats = [], []
        curr = x
        losses = torch.tensor(0.0, device=x.device)

        # 分解过程
        for i in range(self.levels):
            # flat = curr.view(B * C, 1, -1)
            flat = torch.reshape(curr,(B * C, 1, -1))
            # 应用非对称反射填充
            flat_pad = F.pad(flat, self.pad, mode='reflect')
            lo_all = F.conv1d(flat_pad, self.lo, stride=self.stride, padding=0)
            hi_all = F.conv1d(flat_pad, self.hi, stride=self.stride, padding=0)
            Ld = lo_all.size(-1)

            lo = lo_all.view(B, C, -1, Ld)
            hi = hi_all.view(B, C, -1, Ld)
            w = F.softmax(self.filter_w, dim=0).view(1, 1, -1, 1)

            c = self.norm_c((lo * w).sum(2))
            d = self.norm_d((hi * w).sum(2))

            dets.append(d)
            en_feats.append(torch.stack([c.abs().mean(-1), d.abs().mean(-1)], dim=-1))

            losses = losses + self.lambda_d * F.smooth_l1_loss(d, torch.zeros_like(d))
            er = torch.norm(c) / (torch.norm(curr[..., ::2]) + 1e-6) - 1
            losses = losses + self.lambda_c * (er ** 2)
            curr = c

        app = curr

        # 返回分解结果
        return dets, app, en_feats, losses
        # return {
        #     'dets': dets,  # 未应用门控的细节系数列表
        #     'app': app,  # 未应用门控的近似系数
        #     'en_feats': en_feats,  # 能量特征列表
        #     'losses': losses  # 正则损失
        # }

    def reconstruct(self, dets,app,en_feats):
        # 从分解结果中提取数据
        # dets = decomposition_dict['dets']
        # app = decomposition_dict['app']
        # en_feats = decomposition_dict['en_feats']

        B = dets[0].shape[0]  # 获取batch size

        # 计算门控
        e = torch.cat(en_feats, dim=1).view(B, -1)
        g = torch.sigmoid(self.gate(e))

        # 应用门控 - 完全保留原始逻辑
        gated_app = app * g[:, 0].view(-1, 1, 1)
        gated_dets = [dets[i] * g[:, 2 * i + 1].view(-1, 1, 1) for i in range(len(dets))]

        # 逆小波重构
        rec = gated_app
        for i in reversed(range(self.levels)):
            det = gated_dets[i]
            if det.size(-1) != rec.size(-1):
                Lmin = min(det.size(-1), rec.size(-1))
                det = det[..., :Lmin]
                rec = rec[..., :Lmin]
            rec = self.PU[i](rec, det, inverse=True)

        return gated_dets + [gated_app], rec

    # 保持原有forward接口
    def forward(self, x):
        # 执行分解
        dets, app, en_feats, losses = self.decompose(x)

        # 执行重构
        gated_coeffs, rec = self.reconstruct(dets,app,en_feats)

        # 返回与原始格式相同的结果
        return dets, app, en_feats, losses,gated_coeffs, rec
    def _dummy_forward(self, length, batch_size, channel, device):
        """
        用全 1 张量跑一次 forward，返回各级系数序列的 time 维度长度列表
        """
        x = torch.ones(batch_size, channel, length, device=device)
        dets, app, en_feats, losses = self.decompose(x)
        # coeffs 是个 list，list[i].shape == [B, C, L_i]

        l = []
        l.append(app.shape[-1])
        for c in dets:
            l.append(c.shape[-1])
        return l

# ------------------------
# 测试
# ------------------------
if __name__ == '__main__':
    batch_size = 32
    seq_len = 96
    nvar = 7
    level = 3
    pred_len = 96
    x = torch.randn(batch_size, nvar, seq_len)
    # m1 = UnifiedPUFusion(channels=nvar, levels=level,input_length=seq_len,pred_length=pred_len)
    #
    # detail_list, approx, modulated_details, modulated_approx, g = m1.decompose(x)
    # rec = m1.reconstruct(modulated_details, modulated_approx)

    # ------------------
    m2 = UnifiedMultiLevel(channels=nvar, levels=level,input_length=seq_len,pred_length=pred_len)

    coeffs, energy, losses = m2.decompose(x)

    # 执行重构
    gated_coeffs, rec = m2.reconstruct(coeffs, energy)
    # -------------------------
    # m3 = UnifiedFusion(channels=nvar, levels=level,input_length=seq_len,pred_length=pred_len)
    #
    # dets, app, en_feats, losses = m3.decompose(x)
    #
    # # 执行重构
    # gated_coeffs, rec = m3.reconstruct(dets, app, en_feats)

    c = 'end'
    # print("OK shapes & recon error:",
    #       rec1.shape, (rec1 - x).abs().mean().item(),
    #       rec2.shape, (rec2 - x).abs().mean().item(),
    #       rec3.shape, (rec3 - x).abs().mean().item())
