import torch.nn as nn
import torch
import numpy as np

from models.decomp_version19 import LearnableWavelet_new_new
from models.decomp_version21 import  UnifiedMultiLevel, UnifiedFusion
# from models.decomp_version23 import  UnifiedMultiLevel, UnifiedFusion
from models.decomp_version22 import UnifiedMultiLevel_new
from models.enhanceModel import LiftBoostLearnWave, ReConstruction
from models.wavelet_patch_mixer_back_20250627165748 import plot_tensors
from models.xPatch import xPathModel, series_decomp
from utils.RevIN import RevIN
from models.decomposition import Decomposition

class WPMixerCore(nn.Module):
    def __init__(self, 
                 input_length = [], 
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 patch_len = [],
                 patch_stride = [],
                 no_decomposition = [],
                 use_amp = [],
                 configs=None,
                 ):
        
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition 
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        
        self.Decomposition_model = Decomposition(input_length = self.input_length, 
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)
        self.wavelet_decomp_new_new = LearnableWavelet_new_new(levels=self.level, input_length=self.input_length,
                                                               pred_length=self.pred_length, batch_size=self.batch_size,
                                                               channel=self.channel, d_model=self.d_model)
        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series
        # (m+1) number of resolutionBranch
        self.input_w_dim_new = self.wavelet_decomp_new_new.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim_new = self.wavelet_decomp_new_new.pred_w_dim  # list of the length of the predicted coefficient series

        self.input_w_dim_new_mixer_low  = self.input_w_dim_new[0]
        self.input_w_dim_new_mixer_high_list  = self.input_w_dim_new[1:]
        self.input_w_dim_new_mixer_high_list.reverse()
        self.input_w_dim_new_mixer_high_list.append(self.input_length)


        self.pred_w_dim_new_mixer_low  = self.pred_w_dim_new[0]
        self.pred_w_dim_new_mixer_high_list  = self.pred_w_dim_new[1:]
        self.pred_w_dim_new_mixer_high_list.reverse()
        self.pred_w_dim_new_mixer_high_list.append(self.pred_length)


        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                           pred_seq = self.pred_w_dim[i],
                                                           batch_size = self.batch_size,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])
        
        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)

        self.x_path = nn.ModuleList([xPathModel(configs,item)for item in self.input_w_dim ])


        self.x_path_low = xPathModel(configs,self.input_w_dim_new_mixer_low)
        self.x_path_high_list = nn.ModuleList([xPathModel(configs,item)for item in self.input_w_dim_new_mixer_high_list ])

        # self.decomp_lift = LiftBoostLearnWave(
        #     init_wavelet=self.wavelet_name,num_filters=8,levels=self.level,input_length=self.input_length,pred_length=self.pred_length,channel=configs.c_in,device=self.device,batch_size=configs.batch_size)
        # (m+1) number of resolutionBranch
        # self.input_w_dim_lift = self.decomp_lift.input_w_dim  # list of the length of the input coefficient series
        # self.pred_w_dim_lift = self.decomp_lift.pred_w_dim  # list of the length of the predicted coefficient series
        # self.x_path_lift = nn.ModuleList([xPathModel(configs,item)for item in self.input_w_dim_lift ])
        # self.construct = ReConstruction(configs,self.decomp_lift)

        self.m1 = UnifiedMultiLevel_new(channels=self.channel, levels=self.level, pred_length=self.pred_length,
                                    input_length=self.input_length,
                                    batch_size=self.batch_size)

        self.m1_dim_input = self.m1.input_w_dim
        self.m1_dim_pred = self.m1.pred_w_dim

        self.resolutionBranch_m1 = nn.ModuleList([ResolutionBranch(input_seq = self.m1_dim_input[i],
                                                           pred_seq = self.m1_dim_pred[i],
                                                           batch_size = self.batch_size,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride) for i in range(len(self.m1_dim_input))])

        self.x_path_m1 = nn.ModuleList([xPathModel(configs,item)for item in self.m1_dim_input ])



        self.m2 = UnifiedMultiLevel(lambda_d=0.01, lambda_c=0.01,channels=configs.c_in, levels=self.level,input_length=self.input_length,batch_size = self.batch_size,pred_length=self.pred_length,d_model=self.d_model,device=self.device)
        self.m2_dim_input = self.m2.input_w_dim
        self.m2_dim_pred = self.m2.pred_w_dim

        self.resolutionBranch_m2 = nn.ModuleList([ResolutionBranch(input_seq=self.m2_dim_input[i],
                                                                   pred_seq=self.m2_dim_pred[i],
                                                                   batch_size=self.batch_size,
                                                                   channel=self.channel,
                                                                   d_model=self.d_model,
                                                                   dropout=self.dropout,
                                                                   embedding_dropout=self.embedding_dropout,
                                                                   tfactor=self.tfactor,
                                                                   dfactor=self.dfactor,
                                                                   patch_len=self.patch_len,
                                                                   patch_stride=self.patch_stride) for i in
                                                  range(len(self.m2_dim_input))])
        self.x_path_m2 = nn.ModuleList([xPathModel(configs,item)for item in self.m2_dim_input ])

        self.projector = nn.Linear(self.input_length,self.pred_length)
        self.trend_linear = nn.Linear(self.input_length,self.pred_length)
        self.series_decomp = series_decomp()


    def frequency_interpolation(self,x,seq_len,target_len):
        len_ratio = seq_len/target_len
        x = x.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0),x_fft.size(1),target_len//2+1],dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:,:,:seq_len//2+1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2,n=target_len)
        out = out * len_ratio
        return out
    def up_mixer(self,low,high_list):
        out_low,out_low_s,out_low_t = self.x_path_low(low)
        out_low = out_low_s
        out_level_list = [out_low]
        level_list_reverse = high_list.copy()

        level_list_reverse.reverse()

        res_mix_list = []
        out_high = level_list_reverse[0]
        out_low = out_low + out_high
        res_mix_list.append(out_low)

        out_low,out_low_s,out_low_t = self.x_path_high_list[0](out_low)
        out_high = level_list_reverse[1]
        res_mix_list.append(out_low)
        out_low = out_low_s

        out_level_list.append(out_low)
        for i in range(len(level_list_reverse))[1:]:

            # if i < len(level_list_reverse):
            out_high,out_high_s,out_high_t = self.x_path_high_list[i](out_high)  # 224 48 16 (b*nvar,len//2,dim)
            out_high=out_high_s
            # out_high = out_high  # 224 48 16 (b*nvar,len//2,dim)
            out_high_res = self.frequency_interpolation(out_low,
                                                        self.input_w_dim_new_mixer_high_list[i - 1],
                                                        self.input_w_dim_new_mixer_high_list[i]
                                                        )
            # out_high_res = self.resolutionBranch_new_mix_22[i](out_high_res)  # 224 48 16 (b*nvar,len//2,dim)

            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse):
                out_high = level_list_reverse[i + 1]
            # else:
            #     out_high_res = self.frequency_interpolation(out_low,
            #                                                 low_len * (2 ** (i - 1)),
            #                                                 low_len * (2 ** (i))
            #                                                 )
            #
            #     out_low = self.resolutionBranch_new_mix_max(out_high_res)
            out_level_list.append(out_low)
            res_mix_list.append(out_low)
        # 处理最后一维度
        out_high_res = self.frequency_interpolation(out_low,
                                                    self.input_w_dim_new_mixer_high_list[-2],
                                                    self.input_w_dim_new_mixer_high_list[-1]
                                                    )
        out_low,out_low_s,out_low_t = self.x_path_high_list[-1](out_high_res)
        out_low = out_low_s
        out_level_list.append(out_low)
        # res_mix_list.append(out_low)

        out_level_list.reverse()

        return out_level_list, res_mix_list
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
        # x, moving_mean = self.series_decomp(x)


        x = x.transpose(1, 2) # [batch, channel, look_back_length]
        
        # xA: approximation coefficient series, 
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        # ---------------------new version 2025-06-28 10:54:04 --------------------------------------
        # xA, xD,loss  = self.wavelet_decomp_new_new(x)
        # out_list,reconstruct_list  = self.up_mixer(xA,xD)
        #
        # y = out_list[0]
        # ---------------------new version 2025-06-28 10:54:04 --------------------------------------



        # ---------------------new version 2025-06-29 18:46:37 --------------------------------------mse: 0.3821435272693634, mae: 0.39666852355003357


        #
        #
        # # plot_tensors([x] + [approx] + details,'asdasdasd')
        # # plot_tensors([x] + [res] ,'asdasdasd1111111111')
        #
        #
        # coeffs, energy, losses = self.m2.decompose(x)
        # # plot_tensors([x] + [xA] + xD,'old')
        #
        # # yA = self.resolutionBranch[0](xA)
        # yA = coeffs[-1]
        # yA = self.resolutionBranch_m2[0](yA)
        #
        # yD = []
        # for i in range(len(coeffs[0:-1])):
        #     # yD_i = self.resolutionBranch[i + 1](xD[i])
        #     yD_i = coeffs[0:-1][i]
        #     yD_i = self.resolutionBranch_m2[i + 1](yD_i)
        #     yD.append(yD_i)
        # yD.append(yA)
        # _,y = self.m2.reconstruct(yD,energy )
        # plot_tensors([x] + [modulated_approx] + modulated_details,'21_01')
        # plot_tensors([x] + [yA] + yD,'21_03')
        # plot_tensors([x] + [y] ,'21_02')

        # ---------------------new version 2025-06-29 18:46:37 --------------------------------------mse: 0.3821435272693634, mae: 0.39666852355003357

        # ----------------------------------------old version -------------------------------------------

        # xA, xD = self.Decomposition_model.transform(x)
        #
        # # yA = self.resolutionBranch[0](xA)
        # yA = xA
        # yA, yAs, yAt = self.x_path[0](yA)
        # yA = yAs
        # yD = []
        # for i in range(len(xD)):
        #     # yD_i = self.resolutionBranch[i + 1](xD[i])
        #     yD_i = xD[i]
        #     yD_i, yD_is, yD_it = self.x_path[i + 1](yD_i)
        #     yD_i = yD_is
        #     yD.append(yD_i)
        #
        # y = self.Decomposition_model.inv_transform(yA, yD)
        # losses= 0
        # ----------------------------------------old version -------------------------------------------




        # ---------------------new version 2025-06-29 18:36:18 --------------------------------------mse: 0.38442128896713257, mae: 0.40078577399253845


        #
        #
        # # plot_tensors([x] + [approx] + details,'asdasdasd')
        # # plot_tensors([x] + [res] ,'asdasdasd1111111111')
        #
        #
        coeffs, energy, losses = self.m2.decompose(x)
        # plot_tensors([x] + [xA] + xD,'old')
        energy_new = []
        # yA = self.resolutionBranch[0](xA)
        yA = coeffs[-1]
        yA,yAs,yAt = self.x_path_m2[0](yA)
        yA = yAs+yAt
        yD = []
        for i in range(len(coeffs[0:-1])):
            # yD_i = self.resolutionBranch[i + 1](xD[i])
            # energy_new.append(self.energy_linear[i](energy[i])+energy[i])



            yD_i = coeffs[0:-1][i]
            yD_i,yD_is,yD_it = self.x_path_m2[i + 1](yD_i)
            yD_i = yD_is+yD_it
            yD.append(yD_i)
        yD.append(yA)
        _,y = self.m2.reconstruct(yD,energy )
        # _,y2 = self.m2.reconstruct(coeffs,energy )
        #
        # plot_tensors([x] + coeffs,'21_01')
        # plot_tensors([x] + yD,'21_02')
        # plot_tensors([x] + [y] ,'21_03')
        # plot_tensors([x] + [y2] ,'21_04')

        # ---------------------new version 2025-06-29 18:09:35 --------------------------------------
        # y = self.projector(y)








        y = y.transpose(1, 2)

        # moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1)).permute(0,2,1)

        # y  = moving_mean_out + y
        # y = self.projector(y.permute(0,2,1)).permute(0,2,1)
        y = y[:, -self.pred_length:, :] # decomposition output is always even, but pred length can be odd
        xT = self.revin(y, 'denorm')

        return xT,losses * 10


class ResolutionBranch(nn.Module):
    def __init__(self, 
                 input_seq = [],
                 pred_seq = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [], 
                 dfactor = [],
                 patch_len = [],
                 patch_stride = []):
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
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model) # shared among all channels
        self.mixer1 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num,
                            batch_size = self.batch_size,
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.mixer2 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num, 
                            batch_size = self.batch_size, 
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout) 
        self.head = nn.Sequential(nn.Flatten(start_dim = -2 , end_dim = -1),
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
        x_patch  = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch)) 
        
        out =  self.mixer1(x_emb) 
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
        x_new = torch.cat((x, x_padding), dim = -1)
        x_patch = x_new.unfold(dimension = -1, size = self.patch_len, step = self.patch_stride) 
        return x_patch 
        
        
class Mixer(nn.Module):
    def __init__(self, 
                 input_seq = [],
                 out_seq = [], 
                 batch_size = [], 
                 channel = [], 
                 d_model = [],
                 dropout = [],
                 tfactor = [],
                 dfactor = []):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # expansion factor for patch mixer
        self.dfactor = dfactor # expansion factor for embedding mixer
        
        self.tMixer = TokenMixer(input_seq = self.input_seq, batch_size = self.batch_size, channel = self.channel, pred_seq = self.pred_seq, dropout = self.dropout, factor = self.tfactor, d_model = self.d_model)
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
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [], dropout = [], factor = [], d_model = []):
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
    
