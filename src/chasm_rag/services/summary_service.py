from __future__ import annotations

from dataclasses import dataclass

from ..summarization.chasm_pipeline import ChasmPipeline
from ..verification.claim_verifier import ClaimVerification, ClaimVerifier
from ..verification.safety_gate import SafetyGate, ValidatedOutput
from .document_service import ProcessedDocument


@dataclass
class SummaryServiceResult:
    draft: str
    claims: list[ClaimVerification]
    output: ValidatedOutput


def create_summary(document: ProcessedDocument, pipeline: ChasmPipeline | None = None) -> SummaryServiceResult:
    draft = (pipeline or ChasmPipeline()).summarize(document.store).draft
    claims = ClaimVerifier(document.retriever).verify(draft)
    return SummaryServiceResult(draft, claims, SafetyGate().apply(draft, claims, document.store))

