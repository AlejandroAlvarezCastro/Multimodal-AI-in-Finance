from dataclasses import dataclass
from typing import List, Optional
import random
import torch
import os

from .InterventionAnalyzer import InterventionAnalyzer
from .QuestionAnswerAnalizer import QAAnalyzer
from .CoherenceAnalyzer import CoherenceAnalyzer
from .MultiModalEmbeddings import MultimodalEmbeddings

from MULTIMODAL.AUDIO.AudioEmotionAnalzer import AudioEmotionAnalysis
from MULTIMODAL.TEXT.Analyze.TextEmotionAnalyzer import TextEmotionAnalyzer
from MULTIMODAL.VIDEO.VideoEmotionAnalizer import VideoEmotionAnalysis


@dataclass
class EnsembleInterventionAnalyzer:
    sec10k_model_names: List[str]
    qa_analyzer_models: List[str]
    audio_model_name: Optional[str] = None
    text_model_name: Optional[str] = None
    video_model_name: Optional[str] = None
    NUM_EVALUATIONS: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose: int = 1

    def __post_init__(self):
        self.classifiers = [
            InterventionAnalyzer(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.sec10k_model_names]
        
        self.analyzers = [
            QAAnalyzer(model_name=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.qa_analyzer_models]
        
        self.coherence_analyzer = CoherenceAnalyzer(model_name=self.qa_analyzer_models[0]) # Todo: implementar variabilidad intrinseca y extrinseca  
        self.audio_emotion_analyzer = AudioEmotionAnalysis(model_name=self.audio_model_name, device=self.device) if self.audio_model_name else None
        self.text_emotion_analyzer = TextEmotionAnalyzer(model_name=self.text_model_name, device=self.device) if self.text_model_name else None
        self.video_emmotion_analyzer = VideoEmotionAnalysis(mode=self.video_model_name, device=self.device) if self.video_model_name else None

    def ensemble_qa_analysis(self, question: str, answer: str):
        results = []  # (cat, conf, model_name, raw_outputs)
        model_confidences = {}

        self._print_header("Evaluación QA por modelo")

        for analyzer in self.analyzers:
            # analyzer = QAAnalyzer(model_name=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            cat, conf, details = analyzer.get_pred(question, answer)
            if not cat:
                continue
            results.append((cat, conf, analyzer.model_name, details["raw_outputs"]))
            model_confidences[analyzer.model_name] = {
                "Predicted_category": cat,
                "Confidence": round(conf, 2)
            }
            self._print(f"[{analyzer.model_name}] Predicted: {cat} | Confidence: {conf:.1f}%")

        if not results:
            return None, 0.0, model_confidences, {}

        # Combinar confianzas por categoría
        conf_sum = {}
        for cat, conf, *_ in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        final_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(results), 2)

        self._print_header("Resultado QA combinado")
        self._print(f"✅ Final QA prediction: {final_cat} | Confidence: {avg_conf:.1f}%")

        # Elegir modelo con más confianza para esta categoría
        best_models = [r for r in results if r[0] == final_cat]
        best_models_sorted = sorted(best_models, key=lambda x: x[1], reverse=True)
        top_conf = best_models_sorted[0][1]
        top_candidates = [r for r in best_models_sorted if r[1] == top_conf]
        selected_model = random.choice(top_candidates)

        raw_outputs = selected_model[3]
        best_detail = None
        for raw in raw_outputs:
            evaluations = raw.get("evaluations", [])
            for ev in evaluations:
                if ev.get("answered") == final_cat:
                    best_detail = {
                        "answer_summary": ev.get("answer_summary"),
                        "answer_quote": ev.get("answer_quote")
                    }
                    break
            if best_detail:
                break

        return final_cat, avg_conf, model_confidences, best_detail or {}

    def ensemble_predict(self, text: str):
        results = []
        model_confidences = {}  # Para guardar: nombre_modelo -> (categoría, confianza)

        self._print_header("Predicciones individuales")

        for clf in self.classifiers:
            cat, conf = clf.get_pred(text)
            self._print(f"[{clf.model}] Predicted: {cat} | Confidence: {conf:.1f}%")
            results.append((cat, conf))
            model_confidences[clf.model] = {"Predicted_category": cat, "Confidence": round(conf, 2)}

        # Sumar las confianzas por categoría
        conf_sum = {}
        for cat, conf in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        # Obtener la categoría con mayor confianza total
        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(self.classifiers), 2)

        self._print_header("Resultado combinado")
        self._print(f"✅ Final prediction: {best_cat} | Combined confidence: {avg_conf:.1f}%")

        return best_cat, avg_conf, model_confidences

    def initialize_multimodal_model(self, output_csv_path: str, original_path: str):
        path_csv = output_csv_path
        path_json = os.path.join(original_path, "LEVEL_3.json")
        path_audio = os.path.join(original_path, "audio.mp3")  # o el nombre que uses

        if not all(map(os.path.exists, [path_csv, path_json, path_audio])):
            raise FileNotFoundError("Faltan uno o más archivos requeridos para embeddings")

        self.multimodal_embeddings = MultimodalEmbeddings(
            path_csv=path_csv,
            path_json=path_json,
            audio_file_path=path_audio,
            audio_emotion_analyzer=self.audio_emotion_analyzer,
            text_emotion_analyzer=self.text_emotion_analyzer,
            video_emmotion_analyzer=self.video_emmotion_analyzer
        )

        self._print("[INFO] Multimodal model initialized")

    def generate_structured_output(self) -> dict:
        df = self.multimodal_embeddings.sentences_df
        self.multimodal_embeddings.generar_embeddings()

        df = df[df['classification'].isin(['Monologue', 'Question', 'Answer'])].copy()

        result = {
            "monologue_interventions": {},
        }

        # MONOLOGUES
        monologues = df[df["classification"] == "Monologue"]
        for idx, group in monologues.groupby("intervention_id"):
            full_text = " ".join(group["text"])

            result["monologue_interventions"][str(idx)] = {
                "text": full_text,
                "multimodal_embeddings": self._get_multimodal_dict(group)
            }

        # QA PAIRS
        qa_df = df[df["classification"].isin(["Question", "Answer"])]
        for pair_id, pair_group in qa_df.groupby("Pair"):
            if not isinstance(pair_id, str) or not pair_id.startswith("pair_"):
                continue  # 🛑 Saltar pares inválidos
            question_df = pair_group[pair_group["classification"] == "Question"]
            answer_df = pair_group[pair_group["classification"] == "Answer"]

            question_text = " ".join(question_df["text"])
            answer_text = " ".join(answer_df["text"])

            q_cat, q_conf, q_models = self.ensemble_predict(question_text)
            a_cat, a_conf, a_models = self.ensemble_predict(answer_text)

            try:
                qa_cat, qa_conf, qa_models, qa_details = self.ensemble_qa_analysis(question_text, answer_text)
            except:
                print(f"❌ Error processing QA analysis for pair {pair_id}: {question_text} -> {answer_text}")
                qa_cat, qa_conf, qa_models, qa_details = None, 0.0, {}, {}

            coherence_analyses = []
            for mono_id, monologue in result["monologue_interventions"].items():
                try:
                    coh = self.coherence_analyzer.analyze_coherence(monologue["text"], answer_text)
                    coh["monologue_index"] = int(mono_id)
                    coherence_analyses.append(coh)
                except:
                    pass

            result[pair_id] = {
                "Question": question_text,
                "Answer": answer_text,
                "question_classification": {
                    "Predicted_category": q_cat,
                    "Confidence": q_conf,
                    "Model_confidences": q_models
                },
                "answer_classification": {
                    "Predicted_category": a_cat,
                    "Confidence": a_conf,
                    "Model_confidences": a_models
                },
                "qa_response_classification": {
                    "Predicted_category": qa_cat,
                    "Confidence": qa_conf,
                    "Model_confidences": qa_models,
                    "best_model_details": qa_details
                },
                "coherence_analyses": coherence_analyses,
                "multimodal_embeddings": {
                    "question": self._get_multimodal_dict(question_df),
                    "answer": self._get_multimodal_dict(answer_df)
                }
            }

        return result
    
    def _print(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def _print_header(self, title):
        if self.verbose >= 1:
            print(f"\n{'='*10} {title} {'='*10}")

    def _get_multimodal_dict(self, df_subset):
        """
        Genera el diccionario de multimodal_embeddings para una intervención.
        Si no se han calculado embeddings, se pone None en su lugar.
        """
        num_sent = len(df_subset)
        
        if self.audio_emotion_analyzer is not None:
            audio = df_subset["audio_embedding"].tolist()
        else:
            audio = None

        if self.text_emotion_analyzer is not None:
            text = df_subset["text_embedding"].tolist()
        else:
            text = None

        if self.video_emmotion_analyzer is not None:
            video = df_subset["video_embedding"].tolist()
        else:
            video = None

        return {
            "num_sentences": num_sent if (audio or text) else None,
            "audio": audio,
            "text": text,
            "video": video
        }
