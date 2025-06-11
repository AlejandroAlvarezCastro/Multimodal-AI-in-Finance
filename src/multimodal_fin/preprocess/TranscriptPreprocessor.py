from dataclasses import dataclass

import pandas as pd 
import json
import os


@dataclass
class TranscriptPreprocessor:
    section_col: str = "Conf_Section"
    text_col: str = "text"
    qna_key: str = "questions_and_answers"

    def extract_qna_intro(self, json_path: str) -> str | None:
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    qna_text = data.get(self.qna_key)
                    if isinstance(qna_text, str):
                        return qna_text.split(".")[0].strip()
        except Exception as e:
            print(f"[WARNING] Error leyendo {json_path}: {e}")
        return None

    def preprocess(self, csv_path: str, json_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        qna_intro = self.extract_qna_intro(json_path)

        if qna_intro and self.text_col in df.columns:
            match = df[df[self.text_col].str.contains(qna_intro, case=False, na=False)]
            if not match.empty:
                qna_start_index = match.index[0]
                df[self.section_col] = ['prepared_remarks' if i < qna_start_index else 'q_a' for i in df.index]
            else:
                df[self.section_col] = 'prepared_remarks'
        else:
            df[self.section_col] = 'prepared_remarks'

        return df