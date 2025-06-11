import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from multimodal_fin.emotions.audio.AudioEmotionAnalzer import AudioEmotionAnalysis
from multimodal_fin.emotions.video.VideoEmotionAnalizer import VideoEmotionAnalysis
from multimodal_fin.emotions.text.TextEmotionAnalyzer import TextEmotionAnalyzer

from multimodal_fin.analyzers.multimodal.MultiModalEmbeddings import MultimodalEmbeddings


@dataclass
class EmbeddingsExtractor:
    """
    Extrae embeddings multimodales de un CSV de intervenciones.

    - Audio: emociones del audio.
    - Texto: emociones del texto.
    - Video: emociones del vídeo.
    """
    audio_model_name: Optional[str] = None
    text_model_name: Optional[str] = None
    video_model_name: Optional[str] = None
    device: str = 'cpu'
    verbose: int = 1

    def __post_init__(self):
        # Inicializa analizadores de emociones multimodales
        self.audio_emotion = (
            AudioEmotionAnalysis(model_name=self.audio_model_name, device=self.device)
            if self.audio_model_name else None
        )
        self.text_emotion = (
            TextEmotionAnalyzer(model_name=self.text_model_name, device=self.device)
            if self.text_model_name else None
        )
        self.video_emotion = (
            VideoEmotionAnalysis(mode=self.video_model_name, device=self.device)
            if self.video_model_name else None
        )

    def extract(self, csv_path: str, original_dir: str) -> pd.DataFrame:
        """
        Carga el CSV de intervenciones y genera embeddings multimodales.

        Args:
            csv_path: ruta al CSV con intervenciones clasificadas.
            original_dir: carpeta que contiene LEVEL_3.json y archivos multimedia.

        Returns:
            DataFrame con columnas de embeddings: audio_embedding, text_embedding, video_embedding.
        """
        # Rutas necesarias
        path_csv = csv_path
        path_json = os.path.join(original_dir, "LEVEL_3.json")
        path_audio = os.path.join(original_dir, "audio.mp3")

        # Inicializa el módulo de embeddings
        self.multimodal = MultimodalEmbeddings(
            path_csv=path_csv,
            path_json=path_json,
            audio_file_path=path_audio,
            audio_emotion_analyzer=self.audio_emotion,
            text_emotion_analyzer=self.text_emotion,
            video_emmotion_analyzer=self.video_emotion
        )
        if self.verbose:
            print(f"[INFO] Generando embeddings con MultimodalEmbeddings en {original_dir}")

        # Genera embeddings en memoria
        self.multimodal.generar_embeddings()
        # Devuelve DataFrame con embeddings adjuntos
        return self.multimodal.sentences_df