from __future__ import annotations

from dataclasses import dataclass
import re

from ..evidence.evidence_models import EvidenceUnit
from .bm25_index import BM25Index
from .embedding_service import EmbeddingService
from .vector_index import VectorIndex


@dataclass(frozen=True)
class RetrievalResult:
    evidence: EvidenceUnit
    score: float
    channels: tuple[str, ...]


class HybridRetriever:
    """Per-document lexical, semantic, and exact entity retrieval fused with RRF."""
    def __init__(self, evidence: list[EvidenceUnit], embedder: EmbeddingService | None = None) -> None:
        self.evidence = evidence
        self.bm25 = BM25Index(evidence)
        self.embedder = embedder or EmbeddingService()
        self.vector = VectorIndex(evidence, self.embedder)

    def search(self, query: str, k: int = 6, rrf_k: int = 60) -> list[RetrievalResult]:
        rankings = [("bm25", [item for item, _ in self.bm25.search(query, k * 3)]), ("semantic", [item for item, _ in self.vector.search(query, k * 3)])]
        exact_tokens = {token.casefold() for token in re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+(?:/[A-Za-z]+)?", query) if len(token) > 1}
        exact = [item for item in self.evidence if exact_tokens and exact_tokens.intersection(set(re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+(?:/[A-Za-z]+)?", item.original_text.casefold())))]
        rankings.append(("exact", exact))
        scores: dict[str, float] = {}
        channels: dict[str, list[str]] = {}
        lookup = {item.evidence_id: item for item in self.evidence}
        for channel, items in rankings:
            for rank, item in enumerate(items, start=1):
                scores[item.evidence_id] = scores.get(item.evidence_id, 0) + 1 / (rrf_k + rank)
                channels.setdefault(item.evidence_id, []).append(channel)
        ordered = sorted(scores, key=scores.get, reverse=True)[:k]
        return [RetrievalResult(lookup[key], scores[key], tuple(channels[key])) for key in ordered]
