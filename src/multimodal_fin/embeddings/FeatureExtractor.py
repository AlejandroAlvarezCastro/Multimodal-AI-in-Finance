import numpy as np
import torch


class FeatureExtractor:
    def __init__(self, categories_10k=None, qa_categories=None, max_num_coherences=5):
        self.categories_10k = categories_10k or ["MD&A", "Risk Factors", "Business", "Other"]
        self.qa_categories = qa_categories or ["yes", "no", "partially"]
        self.max_num_coherences = max_num_coherences

    def to_onehot(self, value: str, options: list) -> np.ndarray:
        vec = np.zeros(len(options))
        if value in options:
            vec[options.index(value)] = 1.0
        return vec

    def to_onehot_bool(self, value: bool) -> np.ndarray:
        return np.array([0.0, 1.0]) if value else np.array([1.0, 0.0])

    def safe_len(self, emb) -> int:
        if isinstance(emb, list):
            return len(emb)
        if isinstance(emb, dict):
            return max((len(v) for v in emb.values()), default=0)
        return 0

    def get_array_from_embedding(self, emb_data, n_target: int) -> np.ndarray:
        if isinstance(emb_data, list):
            arr = np.array(emb_data)
        elif isinstance(emb_data, dict):
            if not emb_data:
                return np.zeros((n_target, 7))
            for v in emb_data.values():
                arr = np.array(v)
                if arr.ndim == 2 and arr.shape[1] == 7:
                    break
            else:
                return np.zeros((n_target, 7))
        else:
            return np.zeros((n_target, 7))

        if arr.ndim == 1:  # <--- AÑADIR ESTA COMPROBACIÓN
            arr = arr.reshape(0, 7)

        if arr.shape[0] < n_target:
            pad = np.zeros((n_target - arr.shape[0], 7))
            arr = np.vstack([arr, pad])
        return arr[:n_target]

    def extract(self, node) -> (torch.Tensor, torch.Tensor, np.ndarray):
        n_text = self.safe_len(node.text_embeddings)
        n_audio = self.safe_len(node.audio_embeddings)
        n_video = self.safe_len(node.video_embeddings)
        n = max(n_text, n_audio, n_video, 1)

        text = self.get_array_from_embedding(node.text_embeddings, n)
        audio = self.get_array_from_embedding(node.audio_embeddings, n)
        video = self.get_array_from_embedding(node.video_embeddings, n)
        frases = np.concatenate([text, audio, video], axis=1)  # [n, 21]
        mask = np.ones((1, n))  # [1, n]

        frases = torch.tensor(frases, dtype=torch.float32).unsqueeze(0)  # [1, n, 21]
        mask = torch.tensor(mask).bool()  # [1, n]

        meta = []
        if 'classification' in node.metadata:
            cls = node.metadata['classification']
            meta.append(float(cls.get("Confidence", 0)))
            meta.extend(self.to_onehot(cls.get("Predicted_category", "Other"), self.categories_10k))

        if 'qa_response' in node.metadata:
            qa = node.metadata['qa_response']
            meta.append(float(qa.get("Confidence", 0)))
            meta.extend(self.to_onehot(str(qa.get("Predicted_category", "")).lower(), self.qa_categories))

        if 'coherence' in node.metadata:
            for coh in node.metadata['coherence'][:self.max_num_coherences]:
                meta.extend(self.to_onehot_bool(coh.get("consistent", False)))

        expected_size = 1 + len(self.categories_10k) + 1 + len(self.qa_categories) + 2 * self.max_num_coherences
        meta_vec = np.array(meta, dtype=np.float32)
        if len(meta_vec) < expected_size:
            meta_vec = np.pad(meta_vec, (0, expected_size - len(meta_vec)))
        elif len(meta_vec) > expected_size:
            meta_vec = meta_vec[:expected_size]

        return frases, mask, meta_vec
