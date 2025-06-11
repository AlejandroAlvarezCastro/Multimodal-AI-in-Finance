from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Literal, Optional
import pandas as pd
import difflib
import json
import random

from multimodal_fin.analyzers.metadata.Basics import LLMClient, UncertaintyMixin
from multimodal_fin.analyzers.metadata.Prompt_builder import PromptBuilder

class EvaluatedQA(BaseModel):
    question: str
    answered: Literal['yes', 'partially', 'no']
    answer_summary: str = None
    answer_quote: str = None

class InterventionAnalysis(BaseModel):
    intervention: str
    response: str
    evaluations: List[EvaluatedQA]

@dataclass
class QAAnalyzer(UncertaintyMixin):
    model_name: str = "llama3"
    NUM_EVALUATIONS: int = 5

    def __post_init__(self):
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient(self.model_name)

    def analize_qa(self, intervention: str, response: str):
        messages = self.prompt_builder.analize_qa(intervention, response)
        response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())
        return json.loads(response)

    def get_pred(self, question: str, response: str):
        raw_outputs = []
        labels = []

        for _ in range(self.NUM_EVALUATIONS):
            result = self.analize_qa(question, response)
            label = self._extract_best_match_label(question, result)
            if label:
                labels.append(label)
                raw_outputs.append(result)

        # Si no hay ninguna evaluación válida, devuelve None
        if not labels:
            return None, 0.0, {}

        # Calcular resultado y confianza
        final_label, confidence = self.get_result_and_uncertainty(lambda _: labels.pop(0), question, len(raw_outputs))

        return final_label, confidence, {
            "raw_outputs": raw_outputs  # Esto lo usará el Ensemble para elegir detalles del mejor modelo
        }

    def _extract_best_match_label(self, question: str, result: dict) -> Optional[str]:
        evaluations = result.get('evaluations', [])
        if not evaluations:
            return None

        if len(evaluations) == 1:
            return evaluations[0]['answered']

        # Elegir la evaluación con pregunta más parecida
        best_match = max(
            evaluations,
            key=lambda ev: difflib.SequenceMatcher(None, question.lower(), ev.get("question", "").lower()).ratio()
        )
        return best_match.get("answered")
    
        
    def evaluate_qa_model(self, data: list):
        results = []

        for example in data:
            response = example['response']
            for q in example['label']:
                question = q['question']
                true_label = q['answered']
                pred_label = self.get_pred_question(question, response)

                if pred_label is not None:
                    results.append({
                        "question": question,
                        "response": response,
                        "label": true_label,
                        "classification": pred_label
                    })

        return pd.DataFrame(results)
    
    def get_pred_question(self, question: str, response: str) -> Optional[str]:
        try:
            result = self.analize_qa(question, response)
            evaluations = result.get('evaluations', [])

            if not evaluations:
                print("⚠️ No evaluations returned")
                return None

            # Si solo hay una, usarla directamente
            if len(evaluations) == 1:
                return evaluations[0]['answered']

            # Buscar la evaluación más parecida a la pregunta original
            best_match = max(
                evaluations,
                key=lambda ev: difflib.SequenceMatcher(None, question.lower(), ev.get("question", "").lower()).ratio()
            )

            similarity = difflib.SequenceMatcher(None, question.lower(), best_match.get("question", "").lower()).ratio()
            print(f"🔍 Best match similarity: {similarity:.2f} -> '{best_match['question']}'")

            return best_match.get("answered")

        except Exception as e:
            print(f"❌ Error processing question: {question[:30]}... -> {e}")
            return None

