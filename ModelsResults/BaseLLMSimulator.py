from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import time

@dataclass
class BaseLLMSimulator(ABC):
    models: list
    results_path: str
    results_df: pd.DataFrame

    @abstractmethod
    def classify_with_model(self, model_name: str) -> pd.DataFrame:
        """Clasifica los datos usando el modelo especificado."""
        pass

    @abstractmethod
    def evaluate_model(self, model_name: str, classified_df: pd.DataFrame) -> dict:
        """Evalúa el modelo y devuelve las métricas."""
        pass

    @abstractmethod
    def save_and_plot_results(self):
        """Guarda los resultados y genera gráficas."""
        pass

    def run(self):
        for model_name in self.models:
            try:
                print(f"\n▶ Clasificando con modelo: {model_name}")
                start_time = time.time()
                classified_df = self.classify_with_model(model_name)
                elapsed_time = round(time.time() - start_time, 2)
                print(f"⏱ Tiempo empleado para {model_name}: {elapsed_time}s")

                metrics = self.evaluate_model(model_name, classified_df)

                row_data = {
                    "model_name": model_name,
                    "elapsed_time": elapsed_time,
                    **metrics
                }

                self.results_df = pd.concat(
                    [self.results_df, pd.DataFrame([row_data])],
                    ignore_index=True
                )

            except Exception as e:
                print(f"❌ Error con el modelo {model_name}: {e}")
                continue

        self.save_and_plot_results()