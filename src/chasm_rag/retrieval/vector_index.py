from __future__ import annotations

import numpy as np

from ..evidence.evidence_models import EvidenceUnit
from .embedding_service import EmbeddingService


class VectorIndex:
    def __init__(self, evidence: list[EvidenceUnit], embedder: EmbeddingService) -> None:
        self.evidence, self.embedder = evidence, embedder
        self.vectors = embedder.encode([item.normalized_text for item in evidence]) if evidence else np.empty((0, embedder.dimension), dtype=np.float32)

    def search(self, query: str, k: int = 10) -> list[tuple[EvidenceUnit, float]]:
        if not self.evidence:
            return []
        query_vector = self.embedder.encode([query])[0]
        scores = self.vectors @ query_vector
        indexes = np.argsort(-scores)[:k]
        return [(self.evidence[int(index)], float(scores[index])) for index in indexes if scores[index] > 0]

