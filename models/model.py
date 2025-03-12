import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import Permute, Reshape
from utils.RevIN import RevIN

import matplotlib.pyplot as plt
import numpy as np
from models.wavelet_patch_mixer import WPMixerCore

class WPMixerWrapperShortTermForecast(nn.Module):
    def __init__(self,
                 c_in = [], 
                 c_out = [],
                 seq_len = [],
                 out_len = [], 
                d_model = [],  
                dropout = [], 
                embedding_dropout = [],
                device = [],
                batch_size = [],
                tfactor = [],
                dfactor = [],
                wavelet = [],
                level = [],
                patch_len = [],
                stride = [],
                no_decomposition = [],
                use_amp = []):
        super(WPMixerWrapperShortTermForecast, self).__init__()
        self.model = WPMixer(c_in = c_in, c_out = c_out, seq_len = seq_len, out_len = out_len, d_model = d_model,
                            dropout = dropout, embedding_dropout = embedding_dropout, device = device, batch_size = batch_size,
                            tfactor = tfactor, dfactor = dfactor, wavelet = wavelet, level = level, patch_len = patch_len,
                            stride = stride, no_decomposition = no_decomposition,
                            use_amp = use_amp)
        
    def forward(self, x, _unknown1, _unknown2, _unknown3):
        out = self.model(x)
        return out
    

class WPMixer(nn.Module):
    def __init__(self,
                 configs,device):
        
        super(WPMixer, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.channel_in = configs.c_in
        self.channel_out = configs.c_out
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.embedding_dropout = configs.embedding_dropout
        self.batch_size = configs.batch_size # not required now
        self.tfactor = configs.tfactor
        self.dfactor = configs.dfactor
        self.wavelet = configs.wavelet
        self.level = configs.level
        # patch predictior
        self.actual_seq_len = configs.seq_len
        self.no_decomposition = configs.no_decomposition
        self.use_amp = configs.use_amp
        self.device = device
        
        self.wpmixerCore = WPMixerCore(input_length = self.actual_seq_len,
                                                      pred_length = self.pred_len,
                                                      wavelet_name = self.wavelet,
                                                      level = self.level,
                                                      batch_size = self.batch_size,
                                                      channel = self.channel_in, 
                                                      d_model = self.d_model, 
                                                      dropout = self.dropout, 
                                                      embedding_dropout = self.embedding_dropout,
                                                      tfactor = self.tfactor, 
                                                      dfactor = self.dfactor, 
                                                      device = self.device,
                                                      patch_len = self.patch_len, 
                                                      patch_stride = self.stride,
                                                      no_decomposition = self.no_decomposition,
                                                      use_amp = self.use_amp,
                                                        configs=self.configs
                                       )
        
        
    def forward(self, x):
        pred = self.wpmixerCore(x)
        pred = pred[:, :, -self.channel_out:]
        return pred 




if __name__ == '__main__':
    configs = {}

    configs.task_name = 'long_term_forecast'
    configs.data = 'ETTh1'
    configs.seq_len = 96
    configs.pred_len = 96
    configs.d_model = 256
    configs.tfactor = 5
    configs.dfactor = 8
    configs.wavelet = 'db2'
    configs.level = 2
    configs.patch_len = 16
    configs.stride = 8
    configs.batch_size = 256
    configs.learning_rate = 0.000242438
    configs.lradj = 'type3'
    configs.dropout = 0.4
    configs.embedding_dropout = 0.1
    configs.patience = 12
    configs.train_epochs = 15
    configs.use_amp = True