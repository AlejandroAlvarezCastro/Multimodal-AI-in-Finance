import torch
import torch.nn as nn

from multimodal_fin.embeddings.TransformerEncoder import TransformerEncoderLayer

class ConferenceEncoder(nn.Module):
    def __init__(self, device="cpu", input_dim=512, hidden_dim=256, n_heads=4, d_output=512, max_nodes=1000, weights_path=None):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Positional encoding aprendible (1 extra por el [CLS])
        self.pos_embedding = nn.Embedding(max_nodes + 1, input_dim)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2
        )

        self.proj = nn.Linear(input_dim, d_output)
        
        
        # if weights_path is not None:
        #     self.load_state_dict(torch.load(weights_path, map_location=device))
        #     print(f"✅ Pesos cargados desde: {weights_path}")

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            # Borra el peso conflictivo
            if 'pos_embedding.weight' in state_dict:
                del state_dict['pos_embedding.weight']
                # print("⚠️ Se ignoró el peso pos_embedding del checkpoint (incompatibilidad de tamaño).")
            self.load_state_dict(state_dict, strict=False)
            print(f"✅ Pesos cargados desde: {weights_path}")

    def forward(self, node_embeddings: torch.Tensor, return_attn=False):
        """
        node_embeddings: [n_nodes, input_dim]
        """
        n_nodes = node_embeddings.size(0)

        # Preparar token [CLS]
        cls = self.cls_token.expand(1, -1, -1)  # [1, 1, input_dim]
        input_seq = torch.cat([cls, node_embeddings.unsqueeze(0)], dim=1)  # [1, n+1, input_dim]

        # Codificación posicional (incluye posición 0 para [CLS])
        pos_ids = torch.arange(n_nodes + 1, device=input_seq.device).unsqueeze(0)  # [1, n+1]
        pos_emb = self.pos_embedding(pos_ids)  # [1, n+1, input_dim]
        input_seq = input_seq + pos_emb  # [1, n+1, input_dim]

        # Transformer encoder
        out = self.encoder_layer(input_seq)  # [1, n+1, input_dim]

        if return_attn:
            attn_weights = self.encoder_layer.attn_weights  # [1, n_heads, T, T]
            attn_from_cls = attn_weights[0, 0, 0, 1:].detach().cpu().numpy()  # [n_nodes]
            return self.proj(out[:, 0, :]), attn_from_cls  # [1, d_output], [n_nodes]

        return self.proj(out[:, 0, :])  # [d_output]

    # def forward(self, node_embeddings: torch.Tensor, return_attn=False):
    #     """
    #     node_embeddings: [batch_size, n_nodes, input_dim]
    #     """
    #     B, N, D = node_embeddings.shape  # batch size, num_nodes, input_dim

    #     cls = self.cls_token.expand(B, 1, -1)  # [B, 1, input_dim]
    #     input_seq = torch.cat([cls, node_embeddings], dim=1)  # [B, N+1, input_dim]

    #     out = self.encoder_layer(input_seq)  # [B, N+1, input_dim]

    #     if return_attn:
    #         attn_weights = self.encoder_layer.attn_weights  # [B, n_heads, T, T]
    #         attn_from_cls = attn_weights[:, 0, 0, 1:].detach().cpu().numpy()  # [B, N]
    #         return self.proj(out[:, 0, :]), attn_from_cls  # [B, d_output], [B, N]

    #     return self.proj(out[:, 0, :])  # [B, d_output]