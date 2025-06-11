from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from multimodal_fin.analyzers.metadata.InterventionAnalyzer import InterventionAnalyzer
from multimodal_fin.analyzers.metadata.QuestionAnswerAnalizer import QAAnalyzer
from multimodal_fin.analyzers.metadata.CoherenceAnalyzer import CoherenceAnalyzer

@dataclass
class MetadataEnricher:
    """
    Añade metadata sobre temas, QA y coherencia a un DataFrame con embeddings.

    Estructura resultante:
      - monologue_interventions: dict de monólogos completos.
      - pair_X: dict por cada par Q&A con campos:
          question, answer, answered,
          topic classifications, QA classification,
          coherencia, embeddings.
    """
    sec10k_model_names: List[str]
    qa_analyzer_models: List[str]
    num_evaluations: int = 5
    device: str = 'cpu'
    verbose: int = 1

    def __post_init__(self):
        # Clasificadores de tema (sec10k)
        self.topic_classifiers = [
            InterventionAnalyzer(model=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.sec10k_model_names
        ]
        # Analizadores Q&A
        self.qa_analyzers = [
            QAAnalyzer(model_name=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.qa_analyzer_models
        ]
        # Analizador de coherencia
        first_model = self.qa_analyzers[0].model_name if self.qa_analyzers else None
        self.coherence_analyzer = (
            CoherenceAnalyzer(model_name=first_model)
            if first_model else None
        )

    def enrich(self, df: pd.DataFrame, original_dir: Path) -> Dict[str, Any]:
        """
        Genera el dict enriquecido con metadata.
        """
        result: Dict[str, Any] = {"monologue_interventions": {}}

        # 1) Monologues
        monologues = df[df['classification'] == 'Monologue']
        for idx, group in monologues.groupby('intervention_id'):
            text = " ".join(group['text'])
            embeddings = self._get_multimodal_dict(group)
            topic_cat, topic_conf, topic_models = self._classify_topics(text)

            result['monologue_interventions'][str(idx)] = {
                'text': text,
                'multimodal_embeddings': embeddings,
                'topic_classification': {
                    'Predicted_category': topic_cat,
                    'Confidence': topic_conf,
                    'Model_confidences': topic_models
                }
            }

        # 2) QA pairs
        qa_df = df[df['classification'].isin(['Question', 'Answer'])]
        for pair_id, group in qa_df.groupby('Pair'):
            if not isinstance(pair_id, str) or not pair_id.startswith('pair_'):
                continue

            q_group = group[group['classification'] == 'Question']
            a_group = group[group['classification'] == 'Answer']
            question = " ".join(q_group['text'])
            answer = " ".join(a_group['text'])

            # topic
            q_topic = self._classify_topics(question)
            a_topic = self._classify_topics(answer)
            # QA analysis
            qa_cat, qa_conf, qa_models, qa_details = self._analyze_qa_pair(question, answer)
            answered = qa_details.get('answered') if isinstance(qa_details, dict) else None

            # coherence
            coherence = []
            if self.coherence_analyzer:
                for mono_id, mono in result['monologue_interventions'].items():
                    try:
                        coh = self.coherence_analyzer.analyze_coherence(mono['text'], answer)
                        coh['monologue_index'] = int(mono_id)
                        coherence.append(coh)
                    except Exception:
                        continue

            result[pair_id] = {
                'question': question,
                'answer': answer,
                'answered': answered,
                'question_topic_classification': {
                    'Predicted_category': q_topic[0],
                    'Confidence': q_topic[1],
                    'Model_confidences': q_topic[2]
                },
                'answer_topic_classification': {
                    'Predicted_category': a_topic[0],
                    'Confidence': a_topic[1],
                    'Model_confidences': a_topic[2]
                },
                'qa_response_classification': {
                    'Predicted_category': qa_cat,
                    'Confidence': qa_conf,
                    'Model_confidences': qa_models,
                    'details': qa_details
                },
                'coherence_analyses': coherence,
                'multimodal_embeddings': {
                    'question': self._get_multimodal_dict(q_group),
                    'answer': self._get_multimodal_dict(a_group)
                }
            }

    def _classify_topics(self, text: str) -> Any:
        preds = [(clf.get_pred(text)[0], clf.get_pred(text)[1], clf.model)
                 for clf in self.topic_classifiers]
        # sum confidences
        conf_sum: Dict[str, float] = {}
        for cat, conf, _ in preds:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf
        best, total = max(conf_sum.items(), key=lambda x: x[1])
        avg = round(total / len(preds), 2) if preds else 0.0
        model_conf = {model: {'Predicted_category': cat, 'Confidence': round(c, 2)}
                      for cat, c, model in preds}
        return best, avg, model_conf

    def _analyze_qa_pair(self, question: str, answer: str) -> Any:
        results = []
        model_conf: Dict[str, Dict[str, float]] = {}
        for analyzer in self.qa_analyzers:
            cat, conf, details = analyzer.get_pred(question, answer)
            if not cat:
                continue
            results.append((cat, conf, analyzer.model_name, details))
            model_conf[analyzer.model_name] = {'Predicted_category': cat, 'Confidence': round(conf, 2)}
        if not results:
            return None, 0.0, model_conf, {}
        # combine
        conf_sum: Dict[str, float] = {}
        for cat, conf, *_ in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf
        best, total = max(conf_sum.items(), key=lambda x: x[1])
        avg = round(total / len(results), 2)
        detail = next((d for c,_,_,d in results if c == best and isinstance(d, dict)), {})
        return best, avg, model_conf, detail

    def _get_multimodal_dict(self, df_sub: pd.DataFrame) -> Dict[str, Any]:
        return {
            'num_sentences': len(df_sub),
            'audio': df_sub.get('audio_embedding').tolist() if 'audio_embedding' in df_sub else None,
            'text': df_sub.get('text_embedding').tolist() if 'text_embedding' in df_sub else None,
            'video': df_sub.get('video_embedding').tolist() if 'video_embedding' in df_sub else None
        }
