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
    

class NodeEncoderParams(BaseModel):
    meta_dim: int
    n_heads: int
    d_output: int
    weights_path: str

class ConferenceEncoderParams(BaseModel):
    hidden_dim: int
    n_heads: int
    d_output: int
    weights_path: str

class EmbeddingsPipelineSettings(BaseModel):
    node_encoder: NodeEncoderParams
    conference_encoder: ConferenceEncoderParams


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
    embeddings_pipeline: Optional[EmbeddingsPipelineSettings] = None


def load_settings(config_path: str, config_name: str = "default") -> Settings:
    """
    Carga settings desde YAML con dos bloques top-level:
      - conferences_processing
      - embeddings_pipeline
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    # Procesado de conferencias
    text_confs = raw.get('conferences_processing')
    if not isinstance(text_confs, dict) or config_name not in text_confs:
        raise ValueError(f"Sección '{config_name}' no hallada en 'conferences_processing' de {config_path}")
    conf = text_confs[config_name]

    # Carga bloque embeddings_pipeline solo si existe
    emb_confs = raw.get('embeddings_pipeline')
    emb_settings = None
    if isinstance(emb_confs, dict) and config_name in emb_confs:
        section = emb_confs[config_name]
        emb_settings = EmbeddingsPipelineSettings(
            node_encoder=NodeEncoderParams(**section['node_encoder']),
            conference_encoder=ConferenceEncoderParams(**section['conference_encoder'])
        )

    return Settings(
        input_csv_path      = conf['input_csv_path'],
        qa_models           = conf['qa_models'],
        monologue_models    = conf['monologue_models'],
        sec10k_models       = conf['sec10k_models'],
        qa_analyzer_models  = conf['qa_analyzer_models'],
        audio_model         = conf.get('audio_model'),
        text_model          = conf.get('text_model'),
        video_model         = conf.get('video_model'),
        evals               = conf.get('evals', 3),
        device              = conf.get('device', default_device()),
        verbose             = conf.get('verbose', 1),
        embeddings_pipeline = emb_settings
    )