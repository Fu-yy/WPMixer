import torch.nn as nn
import torch
import numpy as np

from models.decomp_version16 import AdvancedWaveletDecomp
from models.decomp_version19 import LearnableWavelet,LearnableWaveletNew,LearnableWavelet_new_new
from models.new_model import LearnableWaveletAE
from models.xPatch import xPathModel
from utils.RevIN import RevIN
from models.decomposition import Decomposition
import matplotlib.pyplot as plt
import time


def plot_tensors(tensor_list, filename):
    plt.figure(figsize=(10, 6))  # 设置画布大小

    # 遍历每个tensor并绘制
    for i, tensor in enumerate(tensor_list):
        tensor = tensor.permute(0,2,1)
        # 将tensor转换为numpy数组（自动处理GPU/CPU设备）
        data = tensor.cpu().detach().numpy()  # 兼容PyTorch张量
        # 如果是其他框架如TensorFlow，使用 data = tensor.numpy()

        plt.plot(data[0, :, -1],
                 label=f'Tensor {i + 1}',  # 自动生成图例标签
                 linestyle='-',  # 实线连接
                 alpha=0.7)  # 半透明效果

    # 添加图表元素
    plt.title('Tensor Line Plots', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)  # 网格线
    plt.legend()  # 显示图例

    # 自动调整布局并显示
    plt.tight_layout()
    plt.savefig(filename + '.png')


class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 configs=None,
                 ):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.configs = configs
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)
        self.decomp = AdvancedWaveletDecomp(channel=self.configs.c_in, level=self.level, time_points=8,filter_length=5,
                                            dim=self.d_model, ode_solver='rk4', ode_step=0.2)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        # self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
        #                                                         pred_seq=self.pred_w_dim[i],
        #                                                         batch_size=self.batch_size,
        #                                                         channel=self.channel,
        #                                                         d_model=self.d_model,
        #                                                         dropout=self.dropout,
        #                                                         embedding_dropout=self.embedding_dropout,
        #                                                         tfactor=self.tfactor,
        #                                                         dfactor=self.dfactor,
        #                                                         patch_len=self.patch_len,
        #                                                         patch_stride=self.patch_stride) for i in
        #                                        range(len(self.input_w_dim))])

        # self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.configs.seq_len,
        #                                                    pred_seq = self.configs.pred_len,
        #                                                    batch_size = self.batch_size,
        #                                                    channel = self.channel,
        #                                                    d_model = self.d_model,
        #                                                    dropout = self.dropout,
        #                                                    embedding_dropout = self.embedding_dropout,
        #                                                    tfactor = self.tfactor,
        #                                                    dfactor = self.dfactor,
        #                                                    patch_len = self.patch_len,
        #                                                    patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])



        self.revin = RevIN(self.channel, eps=1e-5, affine=True, subtract_last=False)
        self.x_path = nn.ModuleList([xPathModel(configs, item) for item in self.input_w_dim])
        self.wavelet_decomp = LearnableWavelet(levels=3)
        self.wavelet_decomp_new = LearnableWaveletNew(levels=3,input_length=self.input_length,
                                                 pred_length=self.pred_length,batch_size=self.batch_size,channel=self.channel,d_model=self.d_model)
        self.wavelet_decomp_new_new = LearnableWavelet_new_new(levels=3,input_length=self.input_length,
                                                 pred_length=self.pred_length,batch_size=self.batch_size,channel=self.channel,d_model=self.d_model)
        # (m+1) number of resolutionBranch
        self.input_w_dim_new = self.wavelet_decomp_new_new.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim_new = self.wavelet_decomp_new_new.pred_w_dim  # list of the length of the predicted coefficient series

        self.resolutionBranch_new = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim_new[i],
                                                                pred_seq=self.pred_w_dim_new[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim_new))])
        init_wavelet = 'db4'
        num_filters = 8
        levels = 3
        batch_size = 2
        channels = 3
        length = 256  # 需要是2^levels的倍数
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.new_model_decomp = LearnableWaveletAE(
            init_wavelet=init_wavelet,
            num_filters=num_filters,
            levels=levels,
            input_length=self.input_length,
            pred_length=self.pred_length,
            batch_size=self.batch_size,
            channel=self.channel,
            d_model=self.d_model,
            device=self.device
        ).to(device)
    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''

        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        # ########################## 新  注释 begin---------------------------------
        # start = time.perf_counter()
        #
        # yl, yh, losses, filters = self.decomp(x)
        # end = time.perf_counter()
        # print(f"精确耗时：{end - start:.6f} 秒")
        # start = time.perf_counter()
        #
        xA, xD = self.Decomposition_model.transform(x)

        recon = self.Decomposition_model.inv_transform(xA, xD)
        plot_tensors([x]+[recon],'OLDRES')

        # end = time.perf_counter()
        # print(f"精确耗时：{end - start:.6f} 秒")
        #
        # plot_tensors([x]+[yl]+yh,'16')
        # plot_tensors([x]+[xA]+xD,'old')
        # start = time.perf_counter()
        #
        # xA_res = self.wavelet_decomp(x)
        # end = time.perf_counter()
        # print(f"精确耗时：{end - start:.6f} 秒")
        # start = time.perf_counter()
        #
        # plot_tensors([x] + xA_res,'19')
        # xA_res_new = self.wavelet_decomp_new(x)
        # end = time.perf_counter()
        # print(f"精确耗时：{end - start:.6f} 秒")
        #
        # plot_tensors([x] + xA_res_new,'19new')
        start = time.perf_counter()
        xA_new_new, xD_new_new  = self.new_model_decomp.decompose(x)
        end = time.perf_counter()
        print(f"精确耗时：{end - start:.6f} 秒")
        plot_tensors([x] + [xA_new_new] + xD_new_new,'19newnewnew')
        # recon = self.decomp.reconstruct(yA, yD, filters)

        start = time.perf_counter()

        xA_new, xD_new = self.wavelet_decomp_new_new(x)
        end = time.perf_counter()
        print(f"精确耗时：{end - start:.6f} 秒")
        plot_tensors([x] + [xA_new] + xD_new,'19newnew')
        res = self.wavelet_decomp_new_new.reconstruct(xA_new, xD_new)
        plot_tensors([x] + [res] ,'19newnew——res')

        yA = self.resolutionBranch_new[0](xA_new)
        yD = []
        for i in range(len(xD_new)):
            yD_i = self.resolutionBranch_new[i + 1](xD_new[i])
            # yD_i = yh[i]
            # yD_i = self.x_path[i + 1](yD_i)
            yD.append(yD_i)

        # recon = self.decomp.reconstruct(yA, yD, filters)
        y = yD

        # ########################## 新  注释 end---------------------------------

        # ########################## 老  注释 begin---------------------------------
        # xA, xD = self.Decomposition_model.transform(x)
        # # plot_tensors([x.permute(0,2,1)],'x')
        # # plot_tensors([xA.permute(0,2,1)],'xA')
        # # plot_tensors([xD[0].permute(0,2,1)],'xD')
        # yA = self.resolutionBranch[0](xA)
        # # yA = xA
        # # yA = self.x_path[0](yA)
        # yD = []
        # for i in range(len(xD)):
        #     yD_i = self.resolutionBranch[i + 1](xD[i])
        #     # yD_i = xD[i]
        #     # yD_i = self.x_path[i + 1](yD_i)
        #     yD.append(yD_i)
        #
        # y = self.Decomposition_model.inv_transform(yA, yD)
        # ########################## 老  注释  end---------------------------------
        y = y.transpose(1, 2)
        y = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd
        xT = self.revin(y, 'denorm')

        return xT


class ResolutionBranch(nn.Module):
    def __init__(self,
                 input_seq=[],
                 pred_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 patch_len=[],
                 patch_stride=[]):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)  # shared among all channels
        self.mixer1 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.mixer2 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))
        self.revin = RevIN(self.channel)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]

        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''

        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)

        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)

        out = self.head(out)
        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


class Mixer(nn.Module):
    def __init__(self,
                 input_seq=[],
                 out_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 tfactor=[],
                 dfactor=[]):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor  # expansion factor for patch mixer
        self.dfactor = dfactor  # expansion factor for embedding mixer

        self.tMixer = TokenMixer(input_seq=self.input_seq, batch_size=self.batch_size, channel=self.channel,
                                 pred_seq=self.pred_seq, dropout=self.dropout, factor=self.tfactor,
                                 d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: output: [Batch, Channel, Patch_number, d_model]

        '''
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x

