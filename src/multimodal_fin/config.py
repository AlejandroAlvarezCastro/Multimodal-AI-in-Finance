"""
Configuration loader and schemas for the multimodal_fin package.
This module uses Pydantic to define and validate pipeline settings loaded from YAML.
"""
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


def default_device() -> str:
    """
    Determine the default compute device based on PyTorch availability.

    Returns:
        str: 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Settings(BaseModel):
    """
    Schema for pipeline settings.

    Attributes:
        input_csv_path (str): Path to a CSV listing conference directories.
        qa_models (List[str]): List of QA model names for classification.
        monologue_models (List[str]): List of monologue model names.
        sec10k_models (List[str]): List of models for 10k-second analysis.
        qa_analyzer_models (List[str]): List of models for QA pair analysis.
        audio_model (Optional[str]): Name of audio embedding model, if enabled.
        text_model (Optional[str]): Name of text embedding model, if enabled.
        video_model (Optional[str]): Name of video embedding model, if enabled.
        evals (int): Number of ensemble evaluations per sample.
        device (str): Compute device identifier ('cpu' or 'cuda').
        verbose (int): Verbosity level for logging and printouts.
    """
    input_csv_path: str = Field(..., description="Path to CSV with conference folders.")
    qa_models: List[str]
    monologue_models: List[str]
    sec10k_models: List[str]
    qa_analyzer_models: List[str]
    audio_model: Optional[str] = None
    text_model: Optional[str] = None
    video_model: Optional[str] = None
    evals: int = Field(3, description="Number of evaluations per ensemble prediction.")
    device: str = Field(default_factory=default_device, description="Compute device: 'cpu' or 'cuda'.")
    verbose: int = Field(1, description="Verbosity level for pipeline output.")


def load_settings(config_path: str, config_name: str = "default") -> Settings:
    """
    Load and validate pipeline settings from a YAML file.
    Expects a top-level 'configs' mapping with named sub-sections.

    Args:
        config_path (str): Path to the YAML configuration file.
        config_name (str): Name of the sub-section under 'configs' to use.

    Returns:
        Settings: Validated settings object.

    Raises:
        ValueError: If 'configs' or the specified section is missing in the YAML.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)

    configs = raw_cfg.get('configs')
    if not isinstance(configs, dict) or config_name not in configs:
        raise ValueError(f"Configuration section '{config_name}' not found in {config_path}.")

    conf = configs[config_name]
    # Determine embedding model names if enabled
    emb = conf.get('embeddings', {})
    audio = emb.get('audio', {}).get('model_name') if emb.get('audio', {}).get('enabled') else None
    text  = emb.get('text', {}).get('model_name')  if emb.get('text', {}).get('enabled')  else None
    video = emb.get('video', {}).get('model_name') if emb.get('video', {}).get('enabled') else None

    # Instantiate and return validated settings
    return Settings(
        input_csv_path      = conf['input_csv_path'],
        qa_models           = conf['qa_models'],
        monologue_models    = conf['monologue_models'],
        sec10k_models       = conf['sec10k_models'],
        qa_analyzer_models  = conf['qa_analyzer_models'],
        audio_model         = audio,
        text_model          = text,
        video_model         = video,
        evals               = conf.get('evals', 3),
        device              = conf.get('device', default_device()),
        verbose             = conf.get('verbose', 1),
    )
