from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelMetrics:
    model_name: str
    results_path: str
    task_name: str = "Classification Task"
    f1_score_type: str = 'macro'

    def __post_init__(self):
        os.makedirs(self.results_path, exist_ok=True)

    def get_additional_data(self, df: pd.DataFrame) -> dict:
        labels = sorted(df['label'].unique())
        conf_matrix = confusion_matrix(df['label'], df['classification'], labels=labels)
        return {'labels': labels, 'conf_matrix': conf_matrix}

    def get_metrics(self, df: pd.DataFrame) -> dict:
        return {
            'accuracy': accuracy_score(df['label'], df['classification']),
            'precision': precision_score(df['label'], df['classification'], average=self.f1_score_type, zero_division=0.0),
            'recall': recall_score(df['label'], df['classification'], average=self.f1_score_type, zero_division=0.0),
            'f1_score': f1_score(df['label'], df['classification'], average=self.f1_score_type, zero_division=0.0)
        }

    def plot_metrics(self, metrics: dict, additional_data: dict):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            additional_data['conf_matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=additional_data['labels'], yticklabels=additional_data['labels'],
            cbar=False, ax=axes[0]
        )
        axes[0].set_title(f"Confusion Matrix | {self.model_name}", fontsize=14)
        axes[0].set_xlabel("Predicted", fontsize=12)
        axes[0].set_ylabel("True", fontsize=12)

        bars = axes[1].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{bar.get_height():.3f}", ha='center', fontsize=10)
        axes[1].set_title(f"Global Metrics | {self.model_name}", fontsize=14)
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        fig.savefig(os.path.join(self.results_path, f'{self.model_name}_metrics.png'), bbox_inches='tight')
        # plt.show(fig)
        # plt.close(fig)

    def plot_f1_vs_time_all_models(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="elapsed_time", y="f1_score", hue="model_name", s=100, alpha=0.7, palette="tab10")

        for _, row in df.iterrows():
            plt.text(row["elapsed_time"], row["f1_score"], row["model_name"], fontsize=9, ha="left", va="center", rotation=30)

        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("F1 Score")
        plt.title(f"F1 Score vs Time - {self.task_name}")
        plt.grid(True)
        plt.legend(title="Model")
        output_path = os.path.join(self.results_path, f"{self.task_name.replace(' ', '_')}_scatter.png")
        plt.savefig(output_path, bbox_inches="tight")
        # plt.show()
        # plt.close()

    def get_results(self, df: pd.DataFrame) -> dict:
        metrics = self.get_metrics(df)
        additional_data = self.get_additional_data(df)
        self.plot_metrics(metrics, additional_data)
        return metrics