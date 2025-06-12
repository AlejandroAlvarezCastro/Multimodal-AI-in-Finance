from pathlib import Path
import typer
import pandas as pd

from multimodal_fin.config import Settings
from multimodal_fin.processors.conference import ConferenceProcessor
from multimodal_fin.embeddings.ConferencePipeline import ConferenceEmbeddingPipeline


class ProcessRunner:
    def __init__(self, settings: Settings):
        self.processor = ConferenceProcessor(settings)

    def run(self) -> None:
        """
        Ejecuta el pipeline completo de texto y multimodal.
        """
        self.processor.run()


class EmbedRunner:
    def __init__(self, settings: Settings):
        emb_cfg = settings.embeddings_pipeline
        if emb_cfg is None:
            raise ValueError("No hay sección `embeddings_pipeline` en la configuración.")
        self.pipeline = ConferenceEmbeddingPipeline(
            node_encoder_params=emb_cfg.node_encoder.model_dump(),
            conference_encoder_params=emb_cfg.conference_encoder.model_dump(),
            device=settings.device,
        )
        # Modo eval de ambos modelos
        self.pipeline.node_encoder.eval()
        self.pipeline.conference_encoder.eval()

    def run(self, json_path: Path = None, json_csv: Path = None) -> None:
        """
        Genera embeddings desde uno o varios JSON.
        """
        # Validación de inputs
        if json_path and json_csv:
            typer.echo("⚠️ Elige solo --json-path o --json-csv, no ambos", err=True)
            raise typer.Exit(1)
        if not (json_path or json_csv):
            typer.echo("⚠️ Debes indicar --json-path o --json-csv", err=True)
            raise typer.Exit(1)

        # Recopila rutas
        if json_path:
            paths = [json_path]
        else:
            df = pd.read_csv(json_csv)
            paths = list(df['Paths'])

        # Genera embeddings
        for p in paths:
            try:
                emb = self.pipeline.generate_embedding(str(p), return_attn=True)
                arr = emb.detach().cpu().numpy().flatten()
                typer.echo(' ')
                typer.echo('-' * 40)
                typer.echo(' ')
                typer.echo(f"📦 {p} → embedding[{len(arr)}]")
                typer.echo(arr)

            except Exception as e:
                typer.echo(f"❌ Error en {p}: {e}", err=True)