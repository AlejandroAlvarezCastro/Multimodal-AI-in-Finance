from pathlib import Path
import yaml
import torch

from multimodal_fin.config import Settings
from multimodal_fin.utils.logging import get_logger
from multimodal_fin.utils.files import read_paths_csv, make_processed_path
from multimodal_fin.processors.Preprocessor import Preprocessor
from multimodal_fin.processors.MultiModalProcessor import MultimodalProcessor

logger = get_logger(__name__)


class ConferenceProcessor:
    """
    Orquesta el flujo completo de procesamiento de conferencias:
      1. Preprocesado de transcripción y separación de secciones.
      2. Clasificación de texto y anotación de pares Q&A.
      3. Extracción de embeddings multimodales.
      4. Enriquecimiento de metadata con LLMs (temas, QA profundo, coherencia).
      5. Persistencia de resultados (CSV y JSON).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pipeline de texto
        self.preprocessor = Preprocessor(
            qa_model_names=settings.qa_models,
            monologue_model_names=settings.monologue_models,
            num_evaluations=settings.evals,
            verbose=settings.verbose
        )

        # Pipeline multimodal dividido en extractor y enricher
        self.multimodal_processor = MultimodalProcessor(
            sec10k_model_names=settings.sec10k_models,
            qa_analyzer_models=settings.qa_analyzer_models,
            audio_model_name=settings.audio_model,
            text_model_name=settings.text_model,
            video_model_name=settings.video_model,
            num_evaluations=settings.evals,
            device=self.device,
            verbose=settings.verbose
        )

    def run(self) -> None:
        """
        Itera sobre cada ruta en el CSV de entrada y lanza su procesamiento.
        """
        for original_path in read_paths_csv(self.settings.input_csv_path):
            try:
                self._process_conference(Path(original_path))
            except Exception as e:
                logger.error(f"Error procesando {original_path}: {e}", exc_info=True)

    def _process_conference(self, original: Path) -> None:
        """
        Procesa una única carpeta de conferencia.

        Pasos:
          1) Preprocesa y clasifica texto, guarda CSV.
          2) Extrae embeddings y enriquece metadata, guarda JSON.
        """
        logger.info(f"Procesando conferencia: {original}")

        # Directorio de salida
        processed_dir = make_processed_path(original)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Rutas de entrada
        transcript_csv = original / "transcript.csv"
        level3_json = original / "LEVEL_3.json"
        if not transcript_csv.exists() or not level3_json.exists():
            raise FileNotFoundError(f"Faltan archivos en {original}")

        # 1) Pipeline de texto y guardado del CSV procesado
        output_csv = processed_dir / "transcript.csv"
        df = self.preprocessor.process_and_save(
            str(transcript_csv),
            str(level3_json),
            str(output_csv)
        )

        # 2) Pipeline multimodal y guardado del JSON enriquecido
        output_json = processed_dir / "transcript.json"
        self.multimodal_processor.process_and_save(
            input_csv_path=str(output_csv),
            original_dir=original,
            output_json_path=str(output_json)
        )

        logger.info(f"Procesamiento finalizado: resultados en {processed_dir}")
