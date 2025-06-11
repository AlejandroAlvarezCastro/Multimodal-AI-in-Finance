from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from multimodal_fin.analyzers.metadata.InterventionAnalyzer import InterventionAnalyzer
from multimodal_fin.analyzers.metadata.QuestionAnswerAnalizer import QAAnalyzer
from multimodal_fin.analyzers.metadata.CoherenceAnalyzer import CoherenceAnalyzer

@dataclass
class MetadataEnricher:
    """
    Añade metadata sobre temas, QA y coherencia a un DataFrame con embeddings.

    - Clasificación de tema (sec10k) por intervención.
    - Análisis QA de pares de pregunta-respuesta.
    - Coherencia entre monólogo y respuestas.
    """
    sec10k_model_names: List[str]
    qa_analyzer_models: List[str]
    num_evaluations: int = 5
    device: str = 'cpu'
    verbose: int = 1

    def __post_init__(self):
        # Clasificadores de tema 10K
        self.topic_classifiers = [
            InterventionAnalyzer(model=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.sec10k_model_names
        ]
        # Analizadores QA
        self.qa_analyzers = [
            QAAnalyzer(model_name=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.qa_analyzer_models
        ]
        # Analizador de coherencia
        first_qa = self.qa_analyzers[0].model_name if self.qa_analyzers else None
        self.coherence_analyzer = (
            CoherenceAnalyzer(model_name=first_qa)
            if first_qa else None
        )

    def enrich(self, df: pd.DataFrame, original_dir: Path) -> dict:
        """
        Genera el dict enriquecido con metadata a partir del DataFrame de embeddings.
        """
        result: dict = {"monologue_interventions": {}}

        # 1) Monólogos: texto completo + embeddings + topic classification
        monologues = df[df['classification'] == 'Monologue']
        for idx, group in monologues.groupby('intervention_id'):
            full_text = " ".join(group['text'])
            embeddings = self._get_multimodal_dict(group)
            topic_cat, topic_conf, topic_models = self._classify_topics(full_text)

            result['monologue_interventions'][str(idx)] = {
                'text': full_text,
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
            q_df = group[group['classification'] == 'Question']
            a_df = group[group['classification'] == 'Answer']
            question_text = " ".join(q_df['text'])
            answer_text = " ".join(a_df['text'])

            # Metadata: temas para pregunta y respuesta
            q_topic = self._classify_topics(question_text)
            a_topic = self._classify_topics(answer_text)
            # Análisis QA profundo
            qa_cat, qa_conf, qa_models, qa_details = self._analyze_qa_pair(question_text, answer_text)
            # Coherencia con cada monólogo
            coherence_analyses = []
            if self.coherence_analyzer:
                for mono_id, mono in result['monologue_interventions'].items():
                    try:
                        coh = self.coherence_analyzer.analyze_coherence(mono['text'], answer_text)
                        coh['monologue_index'] = int(mono_id)
                        coherence_analyses.append(coh)
                    except:
                        pass

            result[pair_id] = {
                'Question': question_text,
                'Answer': answer_text,
                'question_topic_classification': self._format_topic(q_topic),
                'answer_topic_classification': self._format_topic(a_topic),
                'qa_response_classification': {
                    'Predicted_category': qa_cat,
                    'Confidence': qa_conf,
                    'Model_confidences': qa_models,
                    'Details': qa_details
                },
                'coherence_analyses': coherence_analyses,
                'multimodal_embeddings': {
                    'question': self._get_multimodal_dict(q_df),
                    'answer': self._get_multimodal_dict(a_df)
                }
            }

        return result

    def _classify_topics(self, text: str):
        """Clasifica un texto en tema 10K con varios clasificadores ensemble."""
        preds = []
        for clf in self.topic_classifiers:
            cat, conf = clf.get_pred(text)
            preds.append((cat, conf, clf.model))
        # Sumar confianzas por categoría
        conf_sum = {}
        for cat, conf, _ in preds:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf
        best_cat, total = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total / len(self.topic_classifiers), 2)
        # Model confs
        model_conf = {m: {'Predicted_category': c, 'Confidence': round(f,2)} for c,f,m in [(c,f,m) for c,f,m in preds]}
        return best_cat, avg_conf, model_conf

    def _analyze_qa_pair(self, question: str, answer: str):
        """Análisis profundo de pares QA con múltiples modelos."""
        results = []
        model_conf = {}
        for analyzer in self.qa_analyzers:
            cat, conf, details = analyzer.get_pred(question, answer)
            if not cat:
                continue
            results.append((cat, conf, analyzer.model_name, details.get('raw_outputs', [])))
            model_conf[analyzer.model_name] = {
                'Predicted_category': cat,
                'Confidence': round(conf,2)
            }
        if not results:
            return None, 0.0, model_conf, {}
        # Combina confianzas
        conf_sum = {}
        for cat, conf, *_ in results:
            conf_sum[cat] = conf_sum.get(cat,0) + conf
        best_cat, total = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total/len(results),2)
        # Mejores detalles
        best_details = {}
        for cat, _, _, raws in results:
            if cat == best_cat and raws:
                best_details = raws[0]
                break
        return best_cat, avg_conf, model_conf, best_details

    def _get_multimodal_dict(self, df_sub: pd.DataFrame) -> dict:
        """Extrae listas de embeddings de un subset de DataFrame."""
        audio = df_sub.get('audio_embedding').tolist() if 'audio_embedding' in df_sub else None
        text = df_sub.get('text_embedding').tolist() if 'text_embedding' in df_sub else None
        video = df_sub.get('video_embedding').tolist() if 'video_embedding' in df_sub else None
        return {
            'num_sentences': len(df_sub),
            'audio': audio,
            'text': text,
            'video': video
        }

    def _format_topic(self, topic_tuple):
        """Formatea la tupla de tema en dict legible."""
        cat, conf, model_conf = topic_tuple
        return {
            'Predicted_category': cat,
            'Confidence': conf,
            'Model_confidences': model_conf
        }