from __future__ import annotations

from dataclasses import dataclass

from ..evidence.evidence_store import EvidenceStore
from ..preprocessing.chunker import chunk_evidence
from .generator import ExtractiveGenerator, Generator, TransformersGenerator


@dataclass
class SummaryResult:
    draft: str
    intermediate_summaries: list[str]
    source_evidence_ids: list[str]


class ChasmPipeline:
    def __init__(self, generator: Generator | None = None, chunk_tokens: int = 350) -> None:
        if generator is None:
            from config.settings import settings
            generator = TransformersGenerator(settings.generator_model, settings.use_gpu) if settings.generator_mode == "transformers" else ExtractiveGenerator()
        self.generator, self.chunk_tokens = generator, chunk_tokens

    def summarize(self, store: EvidenceStore) -> SummaryResult:
        chunks = chunk_evidence(store.evidence, self.chunk_tokens)
        try:
            partials = [self.generator.generate(chunk.text) for chunk in chunks if chunk.text.strip()]
        except RuntimeError:
            # A missing optional local model must not leave the document unprocessed.
            generator = ExtractiveGenerator()
            partials = [generator.generate(chunk.text) for chunk in chunks if chunk.text.strip()]
        draft = self.generator.generate("\n".join(partials)) if partials else "No extractable source content was available."
        return SummaryResult(draft, partials, [identifier for chunk in chunks for identifier in chunk.evidence_ids])
