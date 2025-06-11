import torch
from torch.utils.data import Dataset
import random
import numpy as np
from anytree import PreOrderIter

from multimodal_fin.speech_tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from multimodal_fin.speech_tree.ConferenceNode import ConferenceNode


class NodeContrastiveDataset(Dataset):
    def __init__(self, json_paths):
        self.nodes = []
        for path in json_paths:
            try:
                builder = ConferenceTreeBuilder(path)
                root_node = builder.build_tree()
                nodes = [n for n in PreOrderIter(root_node) if n.node_type in {"monologue", "question", "answer"}]
                self.nodes.extend(nodes)
            except Exception as e:
                print(f"❌ Error procesando {path}: {e}")

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        view1 = self.augment(node)
        view2 = self.augment(node)
        return view1, view2

    def augment(self, node: ConferenceNode) -> torch.Tensor:
        def sample_modality(mod):
            if not mod or len(mod) == 0:
                return np.zeros((6, 7))  # <-- ya 6 filas por defecto
            mat = np.array(mod)
            if mat.ndim != 2 or mat.shape[1] != 7:
                return np.zeros((6, 7))  # <-- también salida fija
            if mat.shape[0] >= 6:
                idx = sorted(random.sample(range(mat.shape[0]), 6))
                mat = mat[idx]
            else:
                pad_len = 6 - mat.shape[0]
                pad = np.zeros((pad_len, 7))
                mat = np.vstack([mat, pad])
            return mat

        text = sample_modality(node.text_embeddings)
        audio = sample_modality(node.audio_embeddings)
        video = sample_modality(node.video_embeddings) if node.video_embeddings else np.zeros_like(audio)

        frases = np.concatenate([text, audio, video], axis=1)  # [6, 21]
        frases = torch.tensor(frases, dtype=torch.float32)
        return frases
