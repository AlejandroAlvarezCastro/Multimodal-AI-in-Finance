from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import json
import yaml
import time
import torch
import os

from MULTIMODAL.TEXT.Analyze.EnsembleInterventionAnalyzer import EnsembleInterventionAnalyzer
from MULTIMODAL.TEXT.Classify.EnsembleInterventionClassifier import EnsembleInterventionClassifier


@dataclass
class ConferenceProcessor:
    input_csv_path: str
    qa_models: List[str]
    monologue_models: List[str]
    sec10k_models: List[str]
    qa_analyzer_models: List[str]
    audio_model_name: Optional[str] = None
    text_model_name: Optional[str] = None
    video_model_name: Optional[str] = None
    evals: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose: int = 1

    def __post_init__(self):

        self.classifier = EnsembleInterventionClassifier(
            qa_model_names=self.qa_models,
            monologue_model_names=self.monologue_models,
            NUM_EVALUATIONS=self.evals,
            verbose=self.verbose
        )

        self.analyzer = EnsembleInterventionAnalyzer(
            sec10k_model_names=self.sec10k_models,
            qa_analyzer_models=self.qa_analyzer_models,
            audio_model_name=self.audio_model_name,
            text_model_name=self.text_model_name,
            NUM_EVALUATIONS=self.evals,
            device=self.device,
            verbose=self.verbose
        )

    def run(self):

        df_paths = pd.read_csv(self.input_csv_path)
        if 'path' not in df_paths.columns:
            raise ValueError("El CSV debe contener una columna 'path'")

        for original_path in df_paths['path']:
            file_path = os.path.join(original_path, 'transcript.csv')
            json_path = os.path.join(original_path, 'LEVEL_4.json')

            if not os.path.exists(file_path) or not os.path.exists(json_path):
                print(f"[WARN] Saltando: No se encontraron transcript.csv o LEVEL_4.json en {original_path}")
                continue

            # print("[INFO] Processing file with path: ", original_path)
            self._print(f"[INFO] Processing file with path: {original_path}")

            # === Output path ===
            processed_path = original_path.replace("companies", "processed_companies")
            os.makedirs(processed_path, exist_ok=True)

            output_csv_path = os.path.join(processed_path, "transcript.csv")
            output_json_path = os.path.join(processed_path, "transcript.json")

            try:

                start_time = time.time()

                # === Classification ===
                df = self.classifier.preprocessor.preprocess(file_path, json_path)
                df = self.classifier.classify_dataframe(df)
                df = self.classifier.annotate_question_answer_pairs(df)
                df.to_csv(output_csv_path, index=False)

                # === Analysis ===
                self.analyzer.initialize_multimodal_model(output_csv_path, original_path)
                output = self.analyzer.generate_structured_output()

                elapsed_time = time.time() - start_time
                output["processing_time_seconds"] = round(elapsed_time, 2)
                with open(output_json_path, 'w') as f:
                    json.dump(output, f, indent=4)

                self._print(f"[OK] Procesado y guardado en: {processed_path} (Tiempo: {round(elapsed_time, 2)}s)")

            except Exception as e:
                print(f"[ERROR] Fallo en {file_path}: {e}\n")

    
    def _print(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def _print_header(self, title: str):
        if self.verbose >= 1:
            print(f"\n{'='*10} {title} {'='*10}")



def load_config(config_path: str, config_name: str = "default") -> ConferenceProcessor:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    conf = cfg['configs'][config_name]

    # Verifica si se deben usar los modelos de audio y texto
    audio_model = conf['embeddings']['audio']['model_name'] if conf['embeddings']['audio']['enabled'] else None
    text_model = conf['embeddings']['text']['model_name'] if conf['embeddings']['text']['enabled'] else None
    video_model = conf['embeddings']['video']['model_name'] if conf['embeddings']['video']['enabled'] else None

    processor = ConferenceProcessor(
        input_csv_path=conf['input_csv_path'],
        qa_models=conf['qa_models'],
        monologue_models=conf['monologue_models'],
        sec10k_models=conf['sec10k_models'],
        qa_analyzer_models=conf['qa_analyzer_models'],
        audio_model_name=audio_model,
        text_model_name=text_model,
        video_model_name=video_model,
        evals=conf.get('evals', 3),
        device=conf.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=conf.get('verbose', 1)
    )
    return processor