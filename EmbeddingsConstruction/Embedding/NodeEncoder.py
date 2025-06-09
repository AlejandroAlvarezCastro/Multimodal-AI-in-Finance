import torch
import torch.nn as nn

from .SentenceAttentionEncoder import SentenceAttentionEncoder

class NodeEncoder(nn.Module):
    def __init__(self, 
                 device="cpu",
                 input_dim=21,            # 7 (text) + 7 (audio) + 7 (video)
                 hidden_dim=128,          # Atención por frase
                 meta_dim=32,             # Proyección de metadatos
                 d_output=512,            # Embedding final
                 n_heads=4,
                 categories_10k=None,
                 qa_categories=None,
                 weights_path="weights/node_encoder.pt"
                 ):
        super().__init__()

        self.d_output = d_output
        self.meta_dim = meta_dim
        self.categories_10k = categories_10k or ["MD&A", "Risk Factors", "Business", "Other"]
        self.qa_categories = qa_categories or ["yes", "no", "partially"]
        self.max_num_coherences = 5  # número máximo de entradas de coherencia
        self.device = device
        self.weights_path = weights_path


        # Encoder de frases con atención
        self.frase_encoder = SentenceAttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim, n_heads=n_heads)
        
        if weights_path:
            self.frase_encoder.load_state_dict(torch.load(weights_path))
            print(f"✅ Pesos cargados desde: {weights_path}")
        else:
            print("⚠️ No se han cargado pesos preentrenados para frase_encoder")

        # self.frase_encoder.load_state_dict(torch.load(self.weights_path, map_location=self.device))  # o "cpu"
        self.frase_encoder.to(self.device)

        # Proyección de metadatos
        self.meta_proj = nn.Linear(self._get_meta_input_size(), meta_dim)

        # Proyección final
        self.output_proj = nn.Linear(hidden_dim + meta_dim, d_output)

    def _get_meta_input_size(self) -> int:
        return (
            1 + len(self.categories_10k) +    # clasificación 10K
            1 + len(self.qa_categories) +     # respuesta
            2 * self.max_num_coherences       # coherencias booleanas (1-hot de 2)
        )