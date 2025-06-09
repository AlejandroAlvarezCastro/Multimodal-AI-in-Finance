from torch.utils.data import Dataset
import torch

from anytree import PreOrderIter

from EmbeddingsConstruction.Tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from EmbeddingsConstruction.Tree.ConferenceNode import ConferenceNode

# Dataset para contrastive learning a nivel de conferencia
class ConferenceContrastiveDataset(Dataset):
    def __init__(self, json_paths, extractor, node_encoder, conference_encoder, device="cpu"):
        self.json_paths = json_paths
        self.extractor = extractor
        self.node_encoder = node_encoder
        self.conference_encoder = conference_encoder
        self.device = device

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        self.node_encoder.to(self.device)
        path = self.json_paths[idx]
        builder = ConferenceTreeBuilder(path)
        root = builder.build_tree()

        node_embeddings = []
        for node in PreOrderIter(root):
            if node.is_leaf and node.node_type in {"monologue", "question", "answer"}:
                frases, mask, meta_vec = self.extractor.extract(node)
                frase_summary = self.node_encoder.frase_encoder(frases.to(self.device), mask.to(self.device))
                meta_summary = self.node_encoder.meta_proj(
                    torch.tensor(meta_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                combined = torch.cat([frase_summary, meta_summary], dim=-1)
                node_embedding = self.node_encoder.output_proj(combined).squeeze(0)
                node_embeddings.append(node_embedding)

        if not node_embeddings:
            zero = torch.zeros((1, self.node_encoder.d_output), device=self.device)
            return zero, zero.clone()

        stacked = torch.stack(node_embeddings, dim=0).unsqueeze(0)  # [1, n_nodes, 512]
        global_emb = self.conference_encoder(stacked.to(self.device))  # [1, 512]
        return global_emb, global_emb.clone()  # ambas vistas con batch_size = 1

