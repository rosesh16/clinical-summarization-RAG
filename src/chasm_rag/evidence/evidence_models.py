from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from ..ingestion.extraction_models import SourceLocation


class EvidenceType(str, Enum):
    NARRATIVE = "narrative"
    TABLE = "table"
    MEASUREMENT = "measurement"
    INSTRUCTION = "instruction"
    MEDICATION = "medication_mention"
    OTHER = "other"


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"


@dataclass(frozen=True)
class ReferenceRange:
    lower: float | None = None
    upper: float | None = None
    comparison: str | None = None
    threshold: float | None = None
    raw: str | None = None


@dataclass
class Measurement:
    test_name: str
    value: float | None
    unit: str | None
    reference_range: ReferenceRange | None
    reported_flag: str | None
    page: int
    original_text: str
    parsing_status: str = "parsed"
    consistency_warnings: list[str] = field(default_factory=list)
    evidence_id: str | None = None


@dataclass
class EvidenceUnit:
    original_text: str
    normalized_text: str
    document_id: str
    page: int
    evidence_type: EvidenceType
    location: SourceLocation | None = None
    section: str | None = None
    extraction_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    measurement: Measurement | None = None
    evidence_id: str = field(default_factory=lambda: str(uuid4()))

