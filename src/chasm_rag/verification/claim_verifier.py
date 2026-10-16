from __future__ import annotations

from dataclasses import dataclass
import re

from ..evidence.evidence_models import EvidenceUnit, VerificationStatus
from ..retrieval.hybrid_retriever import HybridRetriever
from .claim_extractor import Claim, extract_claims
from .consistency_checks import source_inconsistency
from .numerical_checks import exact_number_support, exact_unit_support


@dataclass
class ClaimVerification:
    claim: Claim
    status: VerificationStatus
    evidence_ids: list[str]
    excerpts: list[str]
    method: str
    warnings: list[str]


def _negated(text: str) -> bool:
    normalized = re.sub(r"\bnot\s+stated\b", "", text, flags=re.I)
    return bool(re.search(r"\b(?:no|not|denies|without|never)\b", normalized, re.I))


class ClaimVerifier:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def verify_claim(self, claim: Claim, candidate_evidence: list[EvidenceUnit] | None = None) -> ClaimVerification:
        evidence = candidate_evidence if candidate_evidence is not None else [result.evidence for result in self.retriever.search(claim.text, k=4)]
        if not evidence:
            return ClaimVerification(claim, VerificationStatus.UNSUPPORTED, [], [], "hybrid retrieval", ["No relevant source evidence found."])
        exact_numbers, number_warning = exact_number_support(claim.text, evidence)
        exact_units, unit_warning = exact_unit_support(claim.text, evidence)
        # A claim must share meaningful terms with an excerpt; similarity/rank alone is insufficient.
        tokens = set(re.findall(r"[a-z]{3,}", claim.text.casefold()))
        matched = [item for item in evidence if len(tokens.intersection(set(re.findall(r"[a-z]{3,}", item.original_text.casefold())))) >= 1]
        negation_conflict = any(_negated(claim.text) != _negated(item.original_text) for item in matched)
        inconsistent_relevant = source_inconsistency(evidence) and ("non-hdl" in claim.text.casefold() or "derived" in claim.text.casefold() or "subtracting hdl" in claim.text.casefold())
        if inconsistent_relevant:
            status = VerificationStatus.SOURCE_INCONSISTENT
            warnings = ["Relevant source measurement is internally inconsistent."]
        elif not exact_numbers or not exact_units:
            status = VerificationStatus.CONTRADICTED
            warnings = [number_warning or unit_warning or "Numerical or unit mismatch."]
        elif negation_conflict:
            status = VerificationStatus.CONTRADICTED
            warnings = ["Negation differs between claim and source evidence."]
        elif matched:
            uncertain = any(item.extraction_confidence is None or item.warnings for item in matched)
            status = VerificationStatus.UNCERTAIN if uncertain else VerificationStatus.SUPPORTED
            warnings = ["Extraction/OCR uncertainty requires review."] if uncertain else []
        else:
            status = VerificationStatus.UNSUPPORTED
            warnings = ["Retrieved text does not substantively support the claim."]
        return ClaimVerification(claim, status, [item.evidence_id for item in evidence], [item.original_text for item in evidence], "deterministic numeric, lexical, and negation checks", warnings)

    def verify(self, text: str, candidate_evidence: list[EvidenceUnit] | None = None) -> list[ClaimVerification]:
        return [self.verify_claim(claim, candidate_evidence) for claim in extract_claims(text)]
