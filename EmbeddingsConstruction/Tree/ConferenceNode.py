from typing import Dict, List, Optional
from dataclasses import dataclass, field

from anytree import NodeMixin

@dataclass
class ConferenceNode(NodeMixin):
    name: str
    node_type: str
    text_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    audio_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    video_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    num_sentences: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.parent = None  # Será asignado luego