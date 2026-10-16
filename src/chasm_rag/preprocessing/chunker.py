from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from ..evidence.evidence_models import EvidenceUnit


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    evidence_ids: tuple[str, ...]
    text: str
    token_count: int


def _tokens(text: str, tokenizer: object | None = None) -> list[str] | list[int]:
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=False)
    return re.findall(r"\w+|[^\w\s]", text)


def chunk_evidence(evidence: Iterable[EvidenceUnit], max_tokens: int = 350, overlap: int = 40, tokenizer: object | None = None) -> list[Chunk]:
    """Structure-safe packing: evidence units (not arbitrary table cells) are atomic."""
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("max_tokens must be positive and overlap smaller than max_tokens")
    chunks: list[Chunk] = []
    current: list[EvidenceUnit] = []
    current_count = 0
    def flush() -> None:
        nonlocal current, current_count
        if current:
            chunks.append(Chunk(f"chunk-{len(chunks) + 1}", tuple(item.evidence_id for item in current), "\n".join(item.original_text for item in current), current_count))
        current, current_count = [], 0
    for item in evidence:
        count = len(_tokens(item.original_text, tokenizer))
        if current and current_count + count > max_tokens:
            previous = current[-1:] if overlap and len(_tokens(current[-1].original_text, tokenizer)) <= overlap else []
            flush()
            current = previous
            current_count = sum(len(_tokens(item.original_text, tokenizer)) for item in current)
        current.append(item)
        current_count += count
    flush()
    return chunks

