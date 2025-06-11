from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel

import pandas as pd
import json

from ..Basics import LLMClient, UncertaintyMixin
from ..Prompt_builder import PromptBuilder

class Category10K(BaseModel):
    category: Literal['Business', 'Risk Factors', 'MD&A', 'Other']

@dataclass
class InterventionAnalyzer(UncertaintyMixin):
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 10

    def __post_init__(self):
        self.llm = LLMClient(self.model)

    def classify_text(self, text: str) -> str:
        messages = PromptBuilder.prompt_10k(text)
        response = self.llm.chat(messages, schema=Category10K.model_json_schema())
        return json.loads((response))['category']

    def explain_other_category(self, text: str) -> str:
        messages = PromptBuilder.explain_why_other(text)
        return self.llm.chat(messages)

    def get_pred(self, text: str):
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(lambda _: predicted_categories.pop(0), text, self.NUM_EVALUATIONS)
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['classification'] = df['text'].apply(lambda text: self.get_pred(text)[0])
        return df