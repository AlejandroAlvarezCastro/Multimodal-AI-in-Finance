"""
Emotion analysis wrappers for different modalities.

This package provides classes to analyze emotional content in audio, text, and video inputs.
Each analyzer offers a uniform interface via the `analyze` method, returning a dict with
predicted emotion labels and confidence scores.
"""

__all__ = [
    "AudioEmotionAnalyzer",
    "TextEmotionAnalyzer",
    "VideoEmotionAnalyzer",
]