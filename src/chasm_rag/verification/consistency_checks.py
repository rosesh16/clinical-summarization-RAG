from __future__ import annotations

from ..evidence.evidence_models import EvidenceUnit


def source_inconsistency(evidence: list[EvidenceUnit]) -> bool:
    return any(item.measurement and item.measurement.consistency_warnings for item in evidence)

