import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout, Linear, Module

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

        self.attn_weights = None  # aquí guardaremos los pesos

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   need_weights=True,
                                                   average_attn_weights=False)
        self.attn_weights = attn_weights  # [batch_size, n_heads, T, T]

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src