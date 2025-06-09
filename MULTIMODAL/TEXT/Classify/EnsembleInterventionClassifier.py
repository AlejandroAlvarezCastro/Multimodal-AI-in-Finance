from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from .QAClassifier import QAClassifier
from .MonologueClassifier import MonologueClassifier
from .TranscriptPreprocessor import TranscriptPreprocessor


@dataclass
class EnsembleInterventionClassifier:
    qa_model_names: List[str]
    monologue_model_names: List[str]
    NUM_EVALUATIONS: int = 5
    verbose: int = 1


    def __post_init__(self):

        self.qna_classifiers = [
            QAClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.qa_model_names]
        
        self.monologue_classifiers = [
            MonologueClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.monologue_model_names]
        
        self.preprocessor = TranscriptPreprocessor()

    def ensemble_predict(self, text: str, classifiers: List) -> Tuple[str, float, List[Tuple[str, str, float]]]:
        individual_preds = []

        self._print_header("Predicciones individuales")

        for clf in classifiers:
            cat, conf = clf.get_pred(text)
            individual_preds.append((clf.model, cat, conf))
            self._print(f"[{clf.model}] Predicted: {cat} | Confidence: {conf:.2f}%")

        # Agrupar confianzas por categoría
        conf_sum = {}
        for _, cat, conf in individual_preds:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(classifiers), 2)

        self._print_header("Resultado combinado")
        self._print(f"✅ Final prediction: {best_cat} | Combined confidence: {avg_conf:.2f}%")
        self._print('\n' + '=' * 100 + '\n')

        return best_cat, avg_conf, individual_preds

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print('')
        df['classification'] = ' '
        df['global_confidence'] = 0.
        df['model_predictions'] = None

        qna_mask = df['Conf_Section'] == 'q_a'
        if qna_mask.any():
            preds = df.loc[qna_mask, 'text'].apply(lambda text: self.ensemble_predict(text, self.qna_classifiers))
            df.loc[qna_mask, 'classification'] = preds.apply(lambda x: x[0])
            df.loc[qna_mask, 'global_confidence'] = preds.apply(lambda x: x[1])
            df.loc[qna_mask, 'model_predictions'] = preds.apply(lambda x: x[2])

        remarks_mask = df['Conf_Section'] == 'prepared_remarks'
        if remarks_mask.any():
            preds = df.loc[remarks_mask, 'text'].apply(lambda text: self.ensemble_predict(text, self.monologue_classifiers))
            df.loc[remarks_mask, 'classification'] = preds.apply(lambda x: x[0])
            df.loc[remarks_mask, 'global_confidence'] = preds.apply(lambda x: x[1])
            df.loc[remarks_mask, 'model_predictions'] = preds.apply(lambda x: x[2])

        return df

    def annotate_question_answer_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        pair_id = 1
        current_question_row = None
        pairs = []

        for index, row in df.iterrows():
            if row['classification'] == "Question":
                current_question_row = index
                pairs.append(None)
            elif row['classification'] == "Answer" and current_question_row is not None:
                pairs[current_question_row] = f"pair_{pair_id}"
                pairs.append(f"pair_{pair_id}")
                pair_id += 1
                current_question_row = None
            else:
                pairs.append(None)

        df['Pair'] = pairs
        pair_counts = df['Pair'].value_counts(dropna=True)
        invalid_pairs = pair_counts[pair_counts != 2]

        if not invalid_pairs.empty:
            raise ValueError(f"Error: The following pairs do not have exactly 2 observations:\n{invalid_pairs.to_dict()}")
        
        df["intervention_id"] = df.index

        return df
    
    def _print(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def _print_header(self, title: str):
        if self.verbose >= 1:
            print(f"\n{'='*10} {title} {'='*10}")