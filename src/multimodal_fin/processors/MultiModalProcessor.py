import yaml
import json
from pathlib import Path

from multimodal_fin.analyzers.multimodal.EmbeddingsExtractor import EmbeddingsExtractor
from multimodal_fin.analyzers.metadata.MetadataEnricher import MetadataEnricher


class MultimodalProcessor:
    """
    Orquesta el análisis multimodal en dos etapas:
      1) Extracción de embeddings multimodales.
      2) Enriquecimiento con metadata (QA analysis, coherencia, temas).
      3) Serialización del JSON enriquecido.

    Sigue el principio de responsabilidad única:
      - EmbeddingsExtractor se encarga solo de generar embeddings.
      - MetadataEnricher se encarga solo de enriquecer esos embeddings con LLMs.
    """
    def __init__(
        self,
        sec10k_model_names: list[str],
        qa_analyzer_models: list[str],
        audio_model_name: str | None = None,
        text_model_name: str | None = None,
        video_model_name: str | None = None,
        num_evaluations: int = 5,
        device: str = 'cpu',
        verbose: int = 1
    ):
        # Componente de embeddings
        self.extractor = EmbeddingsExtractor(
            audio_model_name=audio_model_name,
            text_model_name=text_model_name,
            video_model_name=video_model_name,
            device=device,
            verbose=verbose
        )
        # Componente de metadata
        self.enricher = MetadataEnricher(
            sec10k_model_names=sec10k_model_names,
            qa_analyzer_models=qa_analyzer_models,
            num_evaluations=num_evaluations,
            device=device,
            verbose=verbose
        )

    def process_and_save(
        self,
        input_csv_path: str,
        original_dir: Path,
        output_json_path: str
    ) -> dict:
        """
        Ejecuta pipeline multimodal dividido en dos pasos y guarda JSON.

        Args:
            input_csv_path: ruta al CSV de intervenciones clasificadas.
            original_dir: directorio original (contiene LEVEL_3.json y audio).
            output_json_path: ruta de salida para el JSON enriquecido.

        Returns:
            Diccionario con el resultado enriquecido.
        """
        # 1) Extracción de embeddings
        df_with_embeddings = self.extractor.extract(
            csv_path=input_csv_path,
            original_dir=str(original_dir)
        )

        # 2) Enriquecimiento de metadata
        enriched_result = self.enricher.enrich(
            df=df_with_embeddings,
            original_dir=original_dir
        )

        # 3) Serialización a JSON
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_result, f, ensure_ascii=False, indent=2)

        return enriched_result
