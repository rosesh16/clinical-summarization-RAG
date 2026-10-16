from __future__ import annotations

from collections import Counter
import math
import re

from ..evidence.evidence_models import EvidenceUnit


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w.-]+", text.casefold())


class BM25Index:
    """Small dependency-free BM25 implementation appropriate for per-document indexes."""
    def __init__(self, evidence: list[EvidenceUnit], k1: float = 1.5, b: float = 0.75) -> None:
        self.evidence, self.k1, self.b = evidence, k1, b
        self.docs = [Counter(_terms(item.normalized_text)) for item in evidence]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.df: Counter[str] = Counter(term for doc in self.docs for term in doc)

    def search(self, query: str, k: int = 10) -> list[tuple[EvidenceUnit, float]]:
        query_terms = _terms(query)
        n = len(self.docs)
        scores: list[tuple[int, float]] = []
        for index, document in enumerate(self.docs):
            score = 0.0
            for term in query_terms:
                if term not in document:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                tf = document[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1))
                score += idf * tf * (self.k1 + 1) / denominator
            if score:
                scores.append((index, score))
        return [(self.evidence[index], score) for index, score in sorted(scores, key=lambda value: value[1], reverse=True)[:k]]

