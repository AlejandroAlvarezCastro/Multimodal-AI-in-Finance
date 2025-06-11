from transformers import pipeline

import pandas as pd
import torch

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TextEmotionAnalyzer:
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            framework="pt",
        )
        self.label_map = {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "joy": "happy",
            "neutral": "neutral",
            "sadness": "sad",
            "surprise": "surprise"
        }

    def predict(self, text: str) -> List[Dict[str, float]]:
        """Devuelve la lista de etiquetas con sus probabilidades, mapeando los nombres de emociones."""


        raw_preds = self.classifier([text])[0]
        return [{"label": self.label_map.get(pred["label"], pred["label"]), "score": pred["score"]} for pred in raw_preds]

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Devuelve los 'logits' centrados que pueden interpretarse como embeddings emocionales.
        """
        out = self.classifier([text])[0]
        probs = torch.tensor([item['score'] for item in out])
        logits_rel = torch.log(probs)
        logits_centered = logits_rel - logits_rel.mean()
        return logits_centered
    
    def get_top_emotion(self, text: str) -> str:
        """
        Devuelve la etiqueta emocional principal mapeada al formato estándar.
        """
        predictions = self.classifier([text])[0]
        top_prediction = max(predictions, key=lambda x: x['score'])
        mapped_label = self.label_map.get(top_prediction['label'], top_prediction['label'])
        return mapped_label
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica cada texto del DataFrame y añade una columna con la emoción principal.
        
        Args:
            df: DataFrame con una columna de texto.
            text_column: Nombre de la columna con los textos.
            output_column: Nombre de la nueva columna que contendrá la emoción.
        
        Returns:
            DataFrame con la nueva columna de clasificaciones.
        """
        df['classification'] = df['text'].apply(self.get_top_emotion)
        return df