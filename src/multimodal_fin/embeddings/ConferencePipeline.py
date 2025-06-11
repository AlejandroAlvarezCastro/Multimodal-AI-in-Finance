import torch 
from anytree import PreOrderIter

from multimodal_fin.embeddings.FeatureExtractor import FeatureExtractor
from multimodal_fin.embeddings.ConferenceEncoder import ConferenceEncoder
from multimodal_fin.embeddings.NodeEncoder import NodeEncoder
from multimodal_fin.embeddings.EmbeddingsVisualizer import NodeEmbeddingVisualizer
from multimodal_fin.speech_tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from multimodal_fin.speech_tree.ConferenceTreeVisualizer import ConferenceTreeVisualizer
from multimodal_fin.speech_tree.TreeAttentionVisualizer import TreeAttentionVisualizer


class ConferenceEmbeddingPipeline:
    def __init__(self, node_encoder_params: dict, conference_encoder_params: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.node_encoder = NodeEncoder(self.device, **node_encoder_params).to(self.device)
        self.conference_encoder = ConferenceEncoder(self.device, **conference_encoder_params).to(self.device)
        self.extractor = FeatureExtractor(
            categories_10k=self.node_encoder.categories_10k,
            qa_categories=self.node_encoder.qa_categories,
            max_num_coherences=self.node_encoder.max_num_coherences
        )
        
        
    def generate_embedding(self, json_path: str, return_attn: bool = False):
        builder = ConferenceTreeBuilder(json_path)
        self.root = builder.build_tree()

        self._node_embeddings = []
        self._node_names = []
        self._node_types = []
        self._categories_10k = []

        for node in PreOrderIter(self.root):
            if node.is_leaf and node.node_type in {"monologue", "question", "answer"}:
                frases, mask, meta_vec = self.extractor.extract(node)
                frase_summary = self.node_encoder.frase_encoder(frases.to(self.device), mask.to(self.device))
                meta_summary = self.node_encoder.meta_proj(
                    torch.tensor(meta_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                combined = torch.cat([frase_summary, meta_summary], dim=-1)
                node_embedding = self.node_encoder.output_proj(combined).squeeze(0)
                self._node_embeddings.append(node_embedding)
                self._node_names.append(node.name)
                self._node_types.append(node.node_type)

                if node.node_type == "monologue":
                    self._categories_10k.append("None")
                else:
                    cls = node.metadata.get("classification", {})
                    self._categories_10k.append(cls.get("Predicted_category", "None"))

        if not self._node_embeddings:
            return torch.zeros(self.node_encoder.d_output)

        stacked = torch.stack(self._node_embeddings, dim=0)
        if return_attn:
            conference_embedding, attn_weights = self.conference_encoder(stacked, return_attn=True)
            self._attn_weights = attn_weights
            return conference_embedding
        return self.conference_encoder(stacked)

    def visualize(self, plots: dict = None):
        plots = plots or {}
        visualizer = ConferenceTreeVisualizer(self.root)

        if plots.get("tree_structure", False):
            visualizer.show_text_tree()

        if plots.get("plot", False):
            visualizer.show_networkx_tree()

        if any(plots.get(k, False) for k in ("silhouette", "umap")):
            embedding_visualizer = NodeEmbeddingVisualizer(
                embeddings=self._node_embeddings,
                node_names=self._node_names,
                node_types=self._node_types,
                categories_10k=self._categories_10k
            )

            if plots.get("silhouette", False):
                embedding_visualizer.show_metrics()

            if plots.get("umap", False):
                embedding_visualizer.show_umap()

        if plots.get("attention_tree", False) and hasattr(self, "_attn_weights"):
            attention_viz = TreeAttentionVisualizer(
                root=self.root,
                node_names=self._node_names,
                attn_weights=self._attn_weights
            )
            attention_viz.show()




