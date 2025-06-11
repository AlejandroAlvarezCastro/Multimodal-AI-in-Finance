from dataclasses import dataclass
import os
import json
import pandas as pd

from multimodal_fin.preprocess.EnsembleInterventionClassifier import EnsembleInterventionClassifier

@dataclass
class Preprocessor:
    """
    Orquesta todo el pipeline de:
      1. Preprocesado de transcripción (separa prepared_remarks vs q_a).
      2. Clasificación con ensamblado de modelos Q&A y monólogo.
      3. Anotación de pares pregunta-respuesta.

    Responsabilidad única: cada método hace una tarea concreta.
    """
    # Modelos y parámetros
    qa_model_names: list[str]
    monologue_model_names: list[str]
    num_evaluations: int = 5
    verbose: int = 1
    # Columnas y claves JSON usadas en preprocesado
    section_col: str = "Conf_Section"
    text_col: str = "text"
    qna_key: str = "questions_and_answers"

    def __post_init__(self):
        # Inicializa el clasificador de intervenciones
        self.classifier = EnsembleInterventionClassifier(
            qa_model_names=self.qa_model_names,
            monologue_model_names=self.monologue_model_names,
            NUM_EVALUATIONS=self.num_evaluations,
            verbose=self.verbose
        )

    def extract_qna_intro(self, json_path: str) -> str | None:
        """
        Extrae la primera oración del campo de Q&A en el JSON para marcar sección.
        """
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read() or "{}")
            intro = data.get(self.qna_key)
            if isinstance(intro, str) and intro.strip():
                return intro.split(".")[0].strip()
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Error leyendo {json_path}: {e}")
        return None

    def divide_conference(self, csv_path: str, json_path: str) -> pd.DataFrame:
        """
        Carga CSV de transcripción y asigna sección según presencia de Q&A.
        """
        df = pd.read_csv(csv_path)
        intro = self.extract_qna_intro(json_path)
        if intro and self.text_col in df.columns:
            mask = df[self.text_col].str.contains(intro, case=False, na=False)
            if mask.any():
                start = mask.idxmax()
                df[self.section_col] = [
                    'prepared_remarks' if i < start else 'q_a'
                    for i in df.index
                ]
            else:
                df[self.section_col] = 'prepared_remarks'
        else:
            df[self.section_col] = 'prepared_remarks'
        return df

    def process(self, csv_path: str, json_path: str) -> pd.DataFrame:
        """
        Ejecuta preprocesado, clasificación y anotación de pares Q&A.
        """
        # 1) Preprocesado
        df = self.divide_conference(csv_path, json_path)
        # 2) Clasificación de cada fila
        df = self.classifier.classify_dataframe(df)
        # 3) Anotación de pares Q&A
        df = self.classifier.annotate_question_answer_pairs(df)
        return df

    def process_and_save(self, csv_path: str, json_path: str, output_csv_path: str) -> pd.DataFrame:
        """
        Ejecuta pipeline completo y guarda resultado en CSV.
        """
        df = self.process(csv_path, json_path)
        df.to_csv(output_csv_path, index=False)
        if self.verbose:
            print(f"Resultado guardado en: {output_csv_path}")
        return df