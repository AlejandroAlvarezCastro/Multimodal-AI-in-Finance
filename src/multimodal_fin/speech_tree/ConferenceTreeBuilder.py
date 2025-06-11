import json
import re

from multimodal_fin.speech_tree.ConferenceNode import ConferenceNode

class ConferenceTreeBuilder:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def build_tree(self) -> ConferenceNode:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = ConferenceNode(name="Conference", node_type="root")

        # Añadir monólogos
        for key, monologue in data.get("monologue_interventions", {}).items():
            node = ConferenceNode(
                name=f"Monologue_{key}",
                node_type="monologue",
                text_embeddings=monologue['multimodal_embeddings']['text'],
                audio_embeddings=monologue['multimodal_embeddings']['audio'],
                video_embeddings=monologue['multimodal_embeddings']['video'],
                num_sentences=monologue['multimodal_embeddings']['num_sentences'],
                metadata={"text": monologue.get("text")}
            )
            node.parent = root

        # Añadir pares de pregunta-respuesta
        pair_keys = sorted([k for k in data.keys() if re.match(r"pair_\d+", k)],
                        key=lambda x: int(x.split("_")[1]))

        for pair_key in pair_keys:
            pair = data[pair_key]  

            # Nodo del par sin embeddings
            pair_node = ConferenceNode(name=pair_key, node_type="qa_pair")
            pair_node.parent = root

            # Pregunta
            q_node = ConferenceNode(
                name=f"{pair_key}_Question",
                node_type="question",
                text_embeddings=pair['multimodal_embeddings']['question']['text'],
                audio_embeddings=pair['multimodal_embeddings']['question']['audio'],
                video_embeddings=pair['multimodal_embeddings']['question']['video'],
                num_sentences=pair['multimodal_embeddings']['question']['num_sentences'],   
                metadata={
                    "text": pair.get("Question", {}),
                    "classification": pair.get("question_classification", {})
                }
            )
            q_node.parent = pair_node

            # Respuesta
            a_node = ConferenceNode(
                name=f"{pair_key}_Answer",
                node_type="answer",
                text_embeddings=pair['multimodal_embeddings']['answer']['text'],
                audio_embeddings=pair['multimodal_embeddings']['answer']['audio'],
                video_embeddings=pair['multimodal_embeddings']['answer']['video'],
                num_sentences=pair['multimodal_embeddings']['answer']['num_sentences'], 
                metadata={
                    "text": pair.get("Answer", {}),
                    "classification": pair.get("answer_classification", {}),
                    "qa_response": pair.get("qa_response_classification", {}),
                    "coherence": pair.get("coherence_analyses", [])
                }
            )
            a_node.parent = pair_node

        return root