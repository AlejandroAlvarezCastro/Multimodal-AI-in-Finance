import torch
import torch.nn as nn


class SentenceAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int = 21, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_weights: bool = False):
        """
        x: [B, N, input_dim]
        mask: [B, N]
        """
        B, N, _ = x.shape

        # Proyección
        x = self.input_proj(x)  # [B, N, hidden_dim]

        # Añadir token [CLS]
        cls_token = self.cls_token.expand(B, 1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, hidden_dim]

        # Crear máscara de atención
        if mask is not None:
            mask = torch.cat([torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], dim=1)  # [B, N+1]
            key_padding_mask = ~mask.bool()  # [B, N+1]
        else:
            key_padding_mask = None

        # Self-attention (query=key=value=x)
        attn_output, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # nos da [B, n_heads, tgt_len, src_len]
        )

        # Residual + Norm
        x = self.norm(x + self.dropout(attn_output))  # [B, N+1, hidden_dim]

        # Extraer solo el embedding del token [CLS]
        cls_emb = x[:, 0, :]  # [B, hidden_dim]

        if return_weights:
            # Atención desde el token [CLS] (tgt_idx = 0) hacia las entradas (src_idx)
            # attn_weights: [B, n_heads, 1, N+1] → promediamos sobre heads
            attn_to_tokens = attn_weights[:, :, 0, 1:]  # [B, n_heads, N]
            attn_mean = attn_to_tokens.mean(dim=1)  # [B, N]
            return cls_emb, attn_mean

        return cls_emb  # [B, hidden_dim]