from __future__ import annotations

from dataclasses import dataclass

from ..evidence.evidence_store import EvidenceStore
from ..evidence.evidence_models import VerificationStatus
from ..evidence.measurement_validator import classify_measurement
from ..summarization.summary_composer import (
    conservative_lipid_summary,
    conservative_measurement_summary,
    conservative_thyroid_summary,
    is_lipid_panel,
)
from ..evidence.evidence_models import EvidenceType
from .claim_verifier import ClaimVerification


@dataclass
class ValidatedOutput:
    text: str
    is_validated: bool
    used_fallback: bool
    reason: str | None


class SafetyGate:
    def apply(self, draft: str, claims: list[ClaimVerification], store: EvidenceStore) -> ValidatedOutput:
        blocking = [item for item in claims if item.status in {VerificationStatus.CONTRADICTED, VerificationStatus.UNSUPPORTED, VerificationStatus.SOURCE_INCONSISTENT}]
        omitted_abnormal = [
            item
            for item in store.measurements
            if (
                classify_measurement(item).startswith(("HIGH", "LOW"))
                or (
                    item.reported_flag
                    and item.reported_flag.strip().upper()
                    in {"H", "HIGH", "L", "LOW", "ABNORMAL", "BORDERLINE"}
                )
            )
            and item.test_name.casefold() not in draft.casefold()
        ]
        if not blocking:
            if not omitted_abnormal:
                return ValidatedOutput(draft, True, False, None)
        source_instructions = [item.original_text for item in store.evidence if item.evidence_type == EvidenceType.INSTRUCTION and len(item.original_text.strip()) > 20]
        if is_lipid_panel(store.measurements):
            issue_count = len(blocking) + len(omitted_abnormal)
            return ValidatedOutput(conservative_lipid_summary(store.measurements, source_instructions), False, True, f"{issue_count} verification or abnormal-measurement retention check(s) required the fallback.")
        summary = conservative_thyroid_summary(store.measurements, store.evidence)
        if summary is None:
            summary = conservative_measurement_summary(store.measurements)
        lines = [summary]
        lines.extend(
            [
                "",
                "Human review is needed for any unsupported, conflicting, or uncertain information.",
            ]
        )
        issue_count = len(blocking) + len(omitted_abnormal)
        return ValidatedOutput("\n".join(lines), False, True, f"{issue_count} verification or abnormal-measurement retention check(s) required the fallback.")
