from typing import Dict
from abc import ABC, abstractmethod


class AudioEmotionRecognizer(ABC):
    
    @abstractmethod
    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def get_top_emotion(self, emotion_dict: Dict[str, Dict[str, float]]) -> str:
        pass