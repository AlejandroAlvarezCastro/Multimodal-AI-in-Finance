from dataclasses import dataclass
import pandas as pd

from MULTIMODAL.TEXT.Analyze.InterventionAnalyzer import InterventionAnalyzer
from ..BaseLLMSimulator import BaseLLMSimulator
from ..ModelMetrics import ModelMetrics

@dataclass
class TenKSimulator(BaseLLMSimulator):
    labeled_data_path: str
    task_name: str = "SEC 10K Classification"

    def classify_with_model(self, model_name: str) -> pd.DataFrame:
        classifier = InterventionAnalyzer(model_name, NUM_EVALUATIONS=1)
        df = pd.read_csv(self.labeled_data_path)
        return classifier.classify_dataframe(df)

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