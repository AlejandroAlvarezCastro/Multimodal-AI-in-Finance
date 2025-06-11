from dataclasses import dataclass
from pydantic import BaseModel
from typing import List

import json
import ollama

from multimodal_fin.analyzers.metadata.Prompt_builder import PromptBuilder

class ContradictionDetail(BaseModel):
    monologue_excerpt: str
    response_excerpt: str
    explanation: str


class CoherenceAnalysis(BaseModel):
    topic_covered: bool
    consistent: bool
    summary: str
    contradictions: List[ContradictionDetail]


@dataclass
class CoherenceAnalyzer:
    model_name: str = "llama3"

    def __post_init__(self):
        self.prompt_builder = PromptBuilder()

    def analyze_coherence(self, monologue: str, response: str):
        messages = self.prompt_builder.check_coherence(monologue, response)

        result = ollama.chat(
            model=self.model_name,
            messages=messages,
            format=CoherenceAnalysis.model_json_schema(),
            options={'temperature': 0}
        )

        return json.loads(result.message.content)