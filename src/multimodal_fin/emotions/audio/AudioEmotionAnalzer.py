from dataclasses import dataclass
import pandas as pd
import torch

from multimodal_fin.emotions.audio.Emotion2Vec import Emotion2VecRecognizer

@dataclass
class AudioEmotionAnalysis:
    mode: str = "emotion2vec"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = "iic/emotion2vec_plus_large"

    def __post_init__(self):
        match self.mode:
            case "emotion2vec":
                self.recognizer = Emotion2VecRecognizer(model_name=self.model_name, device=self.device)
            case _:
                raise ValueError("Unsupported mode. Currently only 'emotion2vec' is supported.")

    def analyze_audio(self, audio_path: str) -> str:
        emotion_dict = self.recognizer.predict_from_wav(audio_path)
        return self.change_disgust_fear(self.recognizer.get_top_emotion(emotion_dict))

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with audio file paths.")

        df["classification"] = df["Path"].apply(self.analyze_audio)
        return df
    
    def change_disgust_fear(self, prediction: str) -> str:
        if self.mode in ["emotion2vec"]:
            if prediction == "disgusted":
                return "fearful"
            elif prediction == "fearful":
                return "disgusted"
        return prediction

    def get_embeddings(self, audio_path: str) -> torch.Tensor:
        """
        Dado el path de un audio, devuelve el vector de logits emocionales centrados
        y ordenados como: ['happy', 'neutral', 'surprise', 'disgust', 'anger', 'sadness', 'fear']
        """
        # 1. Obtener predicciones del modelo
        emotion_dict = self.recognizer.predict_from_wav(audio_path)

        # 2. Definir el orden final deseado y el mapeo inverso
        standard_order = ['happy', 'neutral', 'surprise', 'disgust', 'anger', 'sadness', 'fear']
        label_map = {
            'happy': 'happy',
            'neutral': 'neutral',
            'surprised': 'surprise',
            'disgusted': 'disgust',
            'angry': 'anger',
            'sad': 'sadness',
            'fearful': 'fear',
            'other': None  # se ignora
        }
        inverse_map = {v: k for k, v in label_map.items() if v is not None}

        # 3. Extraer probabilidades en el orden deseado
        try:
            ordered_probs = [list(emotion_dict.values())[0][inverse_map[label]] for label in standard_order]
        except KeyError as e:
            raise ValueError(f"Falta una emoción esperada en la predicción: {e}")

        # 4. Logits centrados
        probs_tensor = torch.tensor(ordered_probs)
        logits = torch.log(probs_tensor)
        logits_centered = logits - logits.mean()

        return logits_centered