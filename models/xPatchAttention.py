import torch
from torch import nn

from models.MKan import M_KAN


class PositionalEncoding1D(nn.Module):
    """可学习的一维位置编码"""

    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        # x: [B, L, D]，截取前 L 个位置编码
        return x + self.pos_embed[:, : x.size(1), :]


class TransformerEncoder(nn.Module):
    """简化的 Transformer Encoder Block"""

    def __init__(self, dim, n_heads=1, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        h = self.norm1(x)
        y, _ = self.attn(h, h, h)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionFusion(nn.Module):
    """双流交叉注意力融合"""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, season, trend):
        # season, trend: [B, L, D]
        # 季节查询趋势
        s = self.norm1(season)
        cross1, _ = self.attn1(s, trend, trend)
        # 趋势查询季节
        t = self.norm2(trend)
        cross2, _ = self.attn2(t, season, season)
        return season + cross1, trend + cross2


class AttentionNetwork(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, enc_in):
        super(AttentionNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        order = 3

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        # self.fc1 = M_KAN(input_dim=patch_len,output_dim=self.dim, order=order)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.pos_emb_s = PositionalEncoding1D(self.dim, max_len=self.patch_num)
        self.pos_emb_t = PositionalEncoding1D(seq_len, max_len=self.enc_in)
        self.trans_s = TransformerEncoder(self.dim, n_heads=1, mlp_ratio=4)
        self.trans_t = TransformerEncoder(seq_len, 1, 4)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, seq_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(seq_len * 2, seq_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, seq_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(seq_len * 2)

        self.fc6 = nn.Linear(seq_len * 2, seq_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(seq_len // 2)

        self.fc7 = nn.Linear(seq_len // 2, self.seq_len)
        self.cross_fusion = CrossAttentionFusion(self.seq_len, n_heads=1)
        self.dropout = nn.Dropout(0.1)
        # Streams Concatination
        self.fc8 = nn.Linear(seq_len * 2, pred_len)
        # --- 预测头 ---
        # self.fc8 = nn.Sequential(
        #     nn.LayerNorm(2 * seq_len),
        #     nn.Linear(2 * seq_len, 2 * self.dim),
        #     nn.GELU(),
        #     nn.Linear(2 * self.dim, pred_len)
        # )

        self.p_kan_block = M_KAN(input_dim=self.dim,output_dim=self.dim, order=order)
        self.t_kan_block = M_KAN(input_dim=seq_len,output_dim=seq_len, order=order)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)  # 96 -- 104
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 224 12 16
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)  # 224 12 16 -- 224 12 256
        s = self.gelu1(s)
        s = self.bn1(s)

        s = self.pos_emb_s(s)

        res = s  # 224 12 256
        s = self.trans_s(s)  # [B, patch_num, embed_dim]

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)  # 224 12 16

        # Residual Stream
        res = self.p_kan_block(res)
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.pos_emb_t(t)
        t = self.trans_t(t)  # [B, patch_num, embed_dim]
        t = self.t_kan_block(t)

        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)
        t = torch.reshape(t, (B, C, self.seq_len))  # [Batch, Channel, Output]
        s = torch.reshape(s, (B, C, self.seq_len))  # [Batch, Channel, Output]
        f_s, f_t = self.cross_fusion(s, t)  # [B, L, D], [B, L, D]

        # Streams Concatination
        x = torch.cat((f_t, f_s), dim=-1)
        x = self.fc8(x)

        # Channel concatination
        # x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x, s, t


if __name__ == '__main__':
    net = AttentionNetwork(96, 96, 16, 8, 'end', enc_in=7)
    x = torch.randn(32, 96, 7)
    y = torch.randn(32, 96, 7)
    c = net(x, y)
    d = 'end'