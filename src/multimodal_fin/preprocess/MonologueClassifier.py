from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel

import pandas as pd
import json

from multimodal_fin.analyzers.metadata.Basics import LLMClient, UncertaintyMixin
from multimodal_fin.analyzers.metadata.Prompt_builder import PromptBuilder


class CategoryPresentation(BaseModel):
    category: Literal['Monologue', 'Procedure']

@dataclass
class MonologueClassifier(UncertaintyMixin):
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 5
    output_path: str = 'output'

    def __post_init__(self):
        self.llm = LLMClient(self.model)

    def classify_text(self, text: str) -> str:
        messages = PromptBuilder.prompt_monologue(text)
        response = self.llm.chat(messages, schema=CategoryPresentation.model_json_schema())
        return json.loads(response)['category']

    def get_pred(self, text: str):
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(lambda _: predicted_categories.pop(0), text, n=self.NUM_EVALUATIONS)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['classification'] = df['text'].apply(lambda text: self.get_pred(text)[0])
        return df