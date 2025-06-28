import os

import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
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


def getconfig():
    import torch
    import argparse

    torch.set_printoptions(precision=10)

    parser = argparse.ArgumentParser(description='[WPMixer] Long Sequences Forecasting')

    ''' frequent changing hy.params '''
    parser.add_argument('--model', type=str, required=False, choices=['WPMixer'], default='WPMixer',
                        help='model of experiment')
    parser.add_argument('--task_name', type=str, required=False, choices=['long_term_forecast'],
                        default='long_term_forecast')
    parser.add_argument('--data', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic'],
                        help='dataset')
    parser.add_argument('--use_hyperParam_optim', action='store_true', default=False,
                        help='True: HyperParameters optimization using optuna, False: no optimization')
    parser.add_argument('--no_decomposition', action='store_true', default=False,
                        help='whether to use wavelet decomposition')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--n_jobs', type=int, required=False, choices=[1, 2, 3, 4], default=1,
                        help='number_of_jobs for optuna')
    parser.add_argument('--seed', type=int, required=False, default=42, help='random seed')

    ''' Model Parameters '''
    parser.add_argument('--seq_len', type=int, default=512, help='length of the look back window')
    parser.add_argument('--c_in', type=int, default=7, help='length of the look back window')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    parser.add_argument('--d_model', type=int, default=256, help='embedding dimension')
    parser.add_argument('--tfactor', type=int, default=5, help='expansion factor in the patch mixer')
    parser.add_argument('--dfactor', type=int, default=5, help='expansion factor in the embedding mixer')
    parser.add_argument('--wavelet', type=str, default='db2', help='wavelet type for wavelet transform')
    parser.add_argument('--level', type=int, default=1, help='level for multi-level wavelet decomposition')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch size')
    parser.add_argument('--stride', type=int, default=8, help='Stride')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout for mixer')
    parser.add_argument('--embedding_dropout', type=float, default=0.05, help='dropout for embedding layer')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='pytorch weight decay factor')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')

    ''' Infrequent chaning parameters: Some of these has not used in our model '''
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--cols', type=str, nargs='+', default=None,
                        help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--embed', type=str, default=0)
    parser.add_argument('--loss', type=str, default='smoothL1', choices=['mse', 'smoothL1'])
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')

    ''' Optuna Hyperparameters: if you don't pass the argument, then value form the hyperparameters_optuna.py will be considered as search region'''
    parser.add_argument('--optuna_seq_len', type=int, nargs='+', required=False, default=None,
                        help='Optuna seq length list')
    parser.add_argument('--optuna_lr', type=float, nargs='+', required=False, default=None,
                        help='Optuna lr: first-min, 2nd-max')
    parser.add_argument('--optuna_batch', type=int, nargs='+', required=False, default=None,
                        help='Optuna batch size list')
    parser.add_argument('--optuna_wavelet', type=str, nargs='+', required=False, default=None,
                        help='Optuna wavelet type list')
    parser.add_argument('--optuna_tfactor', type=int, nargs='+', required=False, default=None,
                        help='Optuna tfactor list')
    parser.add_argument('--optuna_dfactor', type=int, nargs='+', required=False, default=None,
                        help='Optuna dfactor list')
    parser.add_argument('--optuna_epochs', type=int, nargs='+', required=False, default=None, help='Optuna epochs list')
    parser.add_argument('--optuna_dropout', type=float, nargs='+', required=False, default=None,
                        help='Optuna dropout list')
    parser.add_argument('--optuna_embedding_dropout', type=float, nargs='+', required=False, default=None,
                        help='Optuna embedding_dropout list')
    parser.add_argument('--optuna_patch_len', type=int, nargs='+', required=False, default=None,
                        help='Optuna patch len list')
    parser.add_argument('--optuna_stride', type=int, nargs='+', required=False, default=None,
                        help='Optuna stride len list')
    parser.add_argument('--optuna_lradj', type=str, nargs='+', required=False, default=None,
                        help='Optuna lr adjustment type list')
    parser.add_argument('--optuna_dmodel', type=int, nargs='+', required=False, default=None, help='Optuna dmodel list')
    parser.add_argument('--optuna_weight_decay', type=float, nargs='+', required=False, default=None,
                        help='Optuna weight_decay list')
    parser.add_argument('--optuna_patience', type=int, nargs='+', required=False, default=None,
                        help='Optuna patience list')
    parser.add_argument('--optuna_level', type=int, nargs='+', required=False, default=None, help='Optuna level list')
    parser.add_argument('--optuna_trial_num', type=int, required=False, default=None, help='Optuna trial number')

    # ---
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    # Moving Average
    parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta')
    parser.add_argument('--device', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')

    args = parser.parse_args()
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1],
                    'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1],
                    'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321],
                        'S': [1, 1], 'MS': [321, 1]},
        'ILI': {'data': 'national_illness.csv', 'root_path': './data/illness/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1],
                'MS': [7, 1]},
        'Solar': {'data': 'solar_AL.txt', 'root_path': './data/solar/', 'T': None, 'M': [137, 137], 'S': [None, None],
                  'MS': [None, None]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.c_in = data_info[args.features][0]
        args.c_out = data_info[args.features][1]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    return args


if __name__ == '__main__':

    folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\checkpoints\finetune_TimeDART_ETTh1_M_il336_ll48_pl720_dm32_df64_nh16_el2_dl1_fc1_dp0.2_hdp0.1_ep10_bs16_lr0.0001_dln_1"
    folder_path = r"E:\MyCode\PyCharm_Code\WPMixer\checkpoints\WPMixer_ETTh1_dec-True_sl96_pl96_dm128_bt32_wvcoif4_tf3_df3_ptl96_stl8_sd42"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\Traffic"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"
    # folder_path = r"E:\模型数据\TimeDART相关数据\TimeDART_version2_file\outputs\pretrain_checkpoints\ETTh1_dln_1"
    if os.path.isdir(folder_path):
        checkpoint_path = os.path.join(folder_path, 'checkpoint.pth')
        # checkpoint_path = os.path.join(folder_path, 'ckpt_best.pth')
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            # state_dictlist = state_dict['model_state_dict']
            state_dictlist = state_dict


    configs = getconfig()

    configs.model = 'WPMixer'
    configs.task_name = 'long_term_forecast'
    configs.data = 'ETTh1'
    configs.seq_len = 96
    configs.pred_len = 96
    configs.d_model = 128
    configs.tfactor = 3
    configs.dfactor = 3
    configs.wavelet = 'coif4'
    configs.level = 3
    configs.patch_len = 8
    configs.stride = 4
    configs.batch_size = 32
    configs.learning_rate = 0.0001
    configs.lradj = 'type3'

    configs.root_path = 'E:\MyCode\PyCharm_Code\WPMixer\data\ETT'
    # configs.data = 'Traffic'
    configs.data_path = 'ETTh1.csv'
    configs.device = 'cpu'

    configs.dropout = 0.1
    configs.embedding_dropout = 0.2
    configs.patience = 5
    configs.train_epochs = 10
    configs.use_amp = True
    train_data, train_loader = data_provider(configs, flag="train")
    vali_data, vali_loader = data_provider(configs, flag="val")
    x = torch.randn(1)
    model = WPMixer(configs,x.device)

    # 处理多GPU训练保存的权重（如果有'module.'前缀）
    # state_dict = {k.replace('module.', ''): v for k, v in state_dictlist.items()}  # 去除前缀
################################################加载map   begin # ----------------------------------------
    # new_pth = model.state_dict()
    # public_dict = {}
    #
    # for k, v in state_dictlist.items():
    #     for kk in new_pth.keys():
    #         if kk in k:
    #             public_dict[kk] = v
    #             break
    # new_pth.update(public_dict)
    # model.load_state_dict(new_pth)
################################################加载map   end # ----------------------------------------

    # 加载权重到模型
    # model.load_state_dict(state_dictlist)

    # 设置为评估模式（固定Dropout和BatchNorm）
    model.eval()
    for i, (batch_x, batch_y) in enumerate(
            train_loader
    ):
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # batch_x= torch.randn(1,336,7)
        # batch_y= torch.randn(16,336,4)
        # x_res= torch.randn(16,336,7)

        # configs.device = batch_x.device

        # mask = torch.ones_like(x)
        # # x_enc 64 336 7 ; x_mark_enc 16 336 4 ； batch_x 16 336 7  mask 64 336 7
        c = model(batch_x)
        d = 'end'

