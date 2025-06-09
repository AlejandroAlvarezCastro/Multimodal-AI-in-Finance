from dataclasses import dataclass
import pandas as pd
import json

from MULTIMODAL.TEXT.Analyze.QuestionAnswerAnalizer import QAAnalyzer
from ..BaseLLMSimulator import BaseLLMSimulator
from ..ModelMetrics import ModelMetrics

@dataclass
class AnsweredSimulator(BaseLLMSimulator):
    labeled_data_path: str
    task_name: str = "Question-Answering"

    def classify_with_model(self, model_name: str) -> pd.DataFrame:
        analyzer = QAAnalyzer(model_name=model_name)
        with open(self.labeled_data_path) as f:
            data = [json.loads(line) for line in f]
        results_df = analyzer.evaluate_qa_model(data)
        return results_df

    def evaluate_model(self, model_name: str, classified_df: pd.DataFrame) -> dict:
        evaluator = ModelMetrics(
            model_name=model_name,
            results_path=self.results_path,
            task_name=self.task_name
        )
        return evaluator.get_results(classified_df)

    def save_and_plot_results(self):
        output_file = f"{self.results_path}/models_metrics.csv"
        self.results_df.to_csv(output_file, index=False)

        evaluator = ModelMetrics(
            model_name="ALL_MODELS",
            results_path=self.results_path,
            task_name=self.task_name
        )
        evaluator.plot_f1_vs_time_all_models(self.results_df)