from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

from ..ingestion.extraction_models import ExtractedDocument, SourceLocation
from ..preprocessing.section_parser import parse_sections
from ..preprocessing.text_normalizer import normalize_for_retrieval
from .evidence_models import EvidenceType, EvidenceUnit, Measurement
from .measurement_parser import (
    parse_lipid_profile_layout,
    parse_fragmented_measurements,
    parse_measurement_fields,
    parse_measurement_line,
    parse_measurements,
    reported_flag_in_text,
)
from .measurement_validator import validate_measurements

_MEDICATION = re.compile(r"\b(?:aspirin|metformin|atorvastatin|lisinopril|amlodipine|insulin|[A-Z][a-z]+(?:cillin|pril|sartan|statin))\b", re.I)
_INSTRUCTION = re.compile(r"\b(?:follow[- ]?up|return|recommend(?:ed)?|advised?|plan)\b", re.I)


@dataclass
class EvidenceStore:
    document_id: str
    evidence: list[EvidenceUnit] = field(default_factory=list)

    @property
    def measurements(self) -> list[Measurement]:
        return [item.measurement for item in self.evidence if item.measurement is not None]

    def by_id(self, evidence_id: str) -> EvidenceUnit | None:
        return next((item for item in self.evidence if item.evidence_id == evidence_id), None)


def _type(line: str) -> EvidenceType:
    if _INSTRUCTION.search(line):
        return EvidenceType.INSTRUCTION
    if _MEDICATION.search(line):
        return EvidenceType.MEDICATION
    return EvidenceType.NARRATIVE



def _measurement_from_table_row(
    row: list[str],
    headers: list[str],
    page: int,
    original_text: str | None = None,
) -> Measurement | None:
    """Parse a laboratory table row when its columns are identifiable."""

    normalize = lambda value: re.sub(
        r"[^a-z0-9]+", " ", value.casefold()
    ).strip()

    header_names = [normalize(value) for value in headers]

    def column(predicate: Callable[[str], bool]) -> int | None:
        return next(
            (
                index for index, header in enumerate(header_names)
                if predicate(header)
            ),
            None,
        )

    name_idx = column(
        lambda header: header in {
            "investigation", "test", "test name", "analyte", "parameter",
            "examination",
        }
    )
    value_idx = column(
        lambda header: header in {
            "result", "value", "result value", "observed value", "finding",
        }
    )
    ref_idx = column(
        lambda header: header in {
            "reference value", "reference range", "ref range", "ref interval",
            "reference interval", "biological reference interval",
            "bio ref interval", "normal range",
        }
        or (
            ("reference" in header or "ref" in header)
            and ("range" in header or "interval" in header or "value" in header)
        )
    )
    unit_idx = column(lambda header: header in {"unit", "units"})
    flag_idx = column(
        lambda header: header in {
            "flag", "status", "result flag", "abnormality", "interpretation",
            "remark", "remarks",
        }
    )
    if name_idx is None:
        name_idx = 0
    if name_idx >= len(row):
        return None

    name = row[name_idx].strip()
    range_pattern = re.compile(
        r"[-+]?\d+(?:\.\d+)?\s*(?:-|to)\s*[-+]?\d+(?:\.\d+)?|"
        r"(?:<=|>=|<|>)\s*[-+]?\d+(?:\.\d+)?",
        re.IGNORECASE,
    )
    if ref_idx is None or ref_idx >= len(row) or not range_pattern.search(row[ref_idx]):
        ref_idx = next(
            (
                index for index, cell in enumerate(row)
                if index != name_idx and range_pattern.search(cell)
            ),
            None,
        )
    if ref_idx is None:
        return None
    if (
        value_idx is None
        or value_idx >= len(row)
        or _numeric_result_cell(row[value_idx]) is None
    ):
        value_idx = next(
            (
                index for index, cell in enumerate(row)
                if index not in {name_idx, ref_idx}
                and _numeric_result_cell(cell) is not None
            ),
            None,
        )
    if value_idx is None:
        return None

    value_cell = row[value_idx].strip()
    value = _numeric_result_cell(value_cell)
    if value is None:
        return None
    reference = row[ref_idx].strip()
    source_flag = reported_flag_in_text(
        row[flag_idx].strip()
        if flag_idx is not None and flag_idx < len(row)
        else None
    )
    source_flag = source_flag or reported_flag_in_text(value_cell)
    if source_flag is None:
        source_flag = next(
            (
                reported_flag_in_text(cell) for cell in row
                if reported_flag_in_text(cell)
            ),
            None,
        )
    occupied = {name_idx, value_idx, ref_idx}
    if flag_idx is not None:
        occupied.add(flag_idx)
    candidates = [
        row[unit_idx].strip()
        if unit_idx is not None and unit_idx < len(row)
        else ""
    ]
    candidates.extend(
        cell.strip() for index, cell in enumerate(row) if index not in occupied
    )
    unit = next(
        (
            candidate for candidate in candidates
            if candidate
            and not _is_reported_flag(candidate)
            and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate)
            and not re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:-|to)\s*[-+]?\d+", candidate, re.I)
            and re.fullmatch(r"[A-Za-zµμ%][A-Za-z0-9µμ/%^.-]*", candidate)
        ),
        None,
    )
    raw_row = original_text or " | ".join(cell.strip() for cell in row)
    return parse_measurement_fields(
        name, value, reference, unit, source_flag, page, raw_row
    )


def _is_reported_flag(value: str) -> bool:
    return value.strip().casefold() in {
        "high", "low", "abnormal", "normal", "h", "l", "borderline",
    }


def _numeric_result_cell(value: str) -> str | None:
    flag = r"(?:HIGH|LOW|ABNORMAL|NORMAL|H|L|BORDERLINE)"
    match = re.fullmatch(
        rf"\s*(?:(?:{flag})\s*)?([-+]?\d+(?:\.\d+)?)"
        rf"(?:\s*(?:{flag}))?\s*",
        value,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _is_reference_recommendation_table(rows: list[list[str]]) -> bool:
    header_text = " ".join(
        cell.casefold()
        for row in rows[:2]
        for cell in row
    )
    compact_header = re.sub(r"[^a-z0-9]+", "", header_text)
    return (
        "nla" in header_text
        and "recommendation" in compact_header
    )


def _is_reference_recommendation_heading(line: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", line.casefold())
    return (
        "nla" in normalized
        and "recommendation" in normalized
        and "2014" in normalized
    )


def _measurement_text_before_reference_table(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        compact_line = re.sub(r"[^a-z0-9]+", "", line.casefold())
        if "nla" not in compact_line or "2014" not in compact_line:
            continue
        context = "".join(lines[index:index + 4]).casefold()
        compact_context = re.sub(r"[^a-z0-9]+", "", context)
        if "recommendation" in compact_context:
            return "".join(lines[:index])
    return text


def _laboratory_result_region(text: str) -> str:
    lines = text.splitlines()
    header_index = next(
        (
            index for index, line in enumerate(lines)
            if all(
                term in line.casefold()
                for term in ("investigation", "result", "reference", "unit")
            )
        ),
        None,
    )
    if header_index is None:
        return _measurement_text_before_reference_table(text)

    end_index = len(lines)
    for index in range(header_index + 1, len(lines)):
        normalized = re.sub(r"\s+", " ", lines[index].strip().casefold())
        if (
            normalized.startswith(("nla", "note", "interpretation", "advice:"))
            or normalized.startswith(
                ("instruments:", "thanks for reference", "medical lab technician",
                 "to check report authenticity", "generated on", "www.")
            )
        ):
            end_index = index
            break
    return "\n".join(lines[header_index + 1:end_index])


def _measurement_signature(measurement: Measurement) -> tuple[object, ...]:
    return (
        measurement.page,
        re.sub(r"[^a-z0-9]+", "", measurement.test_name.casefold()),
        measurement.value,
        measurement.unit.casefold() if measurement.unit else None,
        measurement.reference_range.raw.casefold()
        if measurement.reference_range and measurement.reference_range.raw
        else None,
    )


def _table_rows_for_measurements(
    rows: list[list[str]],
) -> tuple[list[str], list[tuple[list[str], str]]]:
    """Return row-oriented records, transposing field-labelled column tables."""
    if not rows:
        return [], []

    def field_name(value: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        if normalized in {
            "investigation", "test", "test name", "analyte", "parameter",
            "examination",
        }:
            return "name"
        if normalized in {
            "result", "value", "result value", "observed value", "finding",
        }:
            return "value"
        if normalized in {
            "reference value", "reference range", "ref range", "ref interval",
            "reference interval", "biological reference interval",
            "bio ref interval", "normal range",
        } or (
            ("reference" in normalized or "ref" in normalized)
            and ("range" in normalized or "interval" in normalized)
        ):
            return "reference"
        if normalized in {"unit", "units"}:
            return "unit"
        if normalized in {
            "flag", "status", "result flag", "abnormality", "interpretation",
            "remark", "remarks",
        }:
            return "flag"
        return None

    labelled_fields = [
        (index, field_name(row[0]))
        for index, row in enumerate(rows)
        if row and field_name(row[0]) is not None
    ]
    recognized = {field for _, field in labelled_fields}
    if {"name", "value", "reference"}.issubset(recognized) and len(labelled_fields) >= 3:
        max_columns = max(len(row) for row in rows)
        headers = [rows[index][0].strip() for index, _ in labelled_fields]
        transposed: list[tuple[list[str], str]] = []
        for column_index in range(1, max_columns):
            values: list[str] = []
            raw_values: list[str] = []
            for row_index, field in labelled_fields:
                cells = rows[row_index]
                value = cells[column_index].strip() if column_index < len(cells) else ""
                values.append(value)
                raw_values.append(f"{cells[0].strip()}: {value}")
            if any(value for value in values):
                transposed.append((values, " | ".join(raw_values)))
        return headers, transposed

    headers = rows[0]
    return headers, [
        (row, " | ".join(cell.strip() for cell in row))
        for row in rows[1:]
        if any(cell.strip() for cell in row)
    ]


def build_evidence_store(document: ExtractedDocument) -> EvidenceStore:
    evidence: list[EvidenceUnit] = []
    parsed: list[Measurement] = []
    for page in document.pages:
        measurement_text = _laboratory_result_region(page.text)
        line_measurements: dict[str, list[Measurement]] = {}
        for measurement in parse_measurements(measurement_text, page.page):
            line_measurements.setdefault(measurement.original_text.strip(), []).append(
                measurement
            )
        fragmented_measurements = parse_fragmented_measurements(measurement_text, page.page)
        parsed_signatures = {
            _measurement_signature(measurement)
            for measurements in line_measurements.values()
            for measurement in measurements
        }

        for section in parse_sections(page.text):
            offset = 0
            section_lines = section.text.splitlines()
            in_reference_recommendations = False
            for line_index, line in enumerate(section_lines):
                if not line.strip():
                    offset += len(line) + 1
                    continue
                in_reference_recommendations = (
                    in_reference_recommendations
                    or _is_reference_recommendation_heading(line)
                )
                matching = line_measurements.get(line.strip(), [])
                measurement = (
                    matching.pop(0)
                    if matching and not in_reference_recommendations
                    else None
                )
                evidence_type = EvidenceType.MEASUREMENT if measurement else _type(line)
                source_text = line
                # PDF line wrapping may split a source instruction after "for" or
                # "with". Join only its immediate continuation for readable source
                # display; the original page text remains available unchanged.
                if evidence_type == EvidenceType.INSTRUCTION and re.search(r"\b(?:for|with|to)\s*$", line, re.I) and line_index + 1 < len(section_lines):
                    source_text = f"{line} {section_lines[line_index + 1].strip()}"
                unit = EvidenceUnit(source_text, normalize_for_retrieval(source_text), document.document_id, page.page,
                                    evidence_type,
                                    SourceLocation(page.page, offset, offset + len(line)), section.title,
                                    page.confidence, list(page.warnings), measurement)
                if measurement:
                    measurement.evidence_id = unit.evidence_id
                    parsed.append(measurement)
                evidence.append(unit)
                offset += len(line) + 1
        for measurement in fragmented_measurements:
            signature = _measurement_signature(measurement)
            if signature in parsed_signatures:
                continue
            parsed_signatures.add(signature)
            source_start = page.text.find(measurement.original_text.splitlines()[0])
            location = SourceLocation(
                page.page,
                source_start if source_start >= 0 else None,
                source_start + len(measurement.original_text)
                if source_start >= 0
                else None,
            )
            unit = EvidenceUnit(
                measurement.original_text,
                normalize_for_retrieval(measurement.original_text),
                document.document_id,
                page.page,
                EvidenceType.MEASUREMENT,
                location=location,
                section="Reconstructed table row",
                extraction_confidence=page.confidence,
                warnings=list(page.warnings),
                measurement=measurement,
            )
            measurement.evidence_id = unit.evidence_id
            parsed.append(measurement)
            evidence.append(unit)
        for table in page.tables:
            if not table.rows:
                continue

            headers, table_data_rows = _table_rows_for_measurements(table.rows)
            reference_recommendations = _is_reference_recommendation_table(
                table.rows
            )

            for row, row_text in table_data_rows:
                row_text = row_text.strip()
                if not row_text:
                    continue

                measurement = (
                    None
                    if reference_recommendations
                    else _measurement_from_table_row(
                        row, headers, page.page, original_text=row_text
                    )
                )
                if measurement is not None:
                    signature = _measurement_signature(measurement)
                    if signature in parsed_signatures:
                        measurement = None
                    else:
                        parsed_signatures.add(signature)

                existing = next(
                    (
                        item for item in evidence
                        if item.page == page.page
                        and item.original_text == row_text
                    ),
                    None,
                )

                if existing is not None:
                    # Attach a measurement to an existing row if it was
                    # previously stored as table evidence without one.
                    if measurement is not None and existing.measurement is None:
                        existing.measurement = measurement
                        existing.evidence_type = EvidenceType.MEASUREMENT
                        measurement.evidence_id = existing.evidence_id
                        parsed.append(measurement)
                    continue

                unit = EvidenceUnit(
                    row_text,
                    normalize_for_retrieval(row_text),
                    document.document_id,
                    page.page,
                    EvidenceType.MEASUREMENT if measurement else EvidenceType.TABLE,
                    section=table.name,
                    extraction_confidence=table.confidence or page.confidence,
                    warnings=list(page.warnings),
                    measurement=measurement,
                )

                if measurement is not None:
                    measurement.evidence_id = unit.evidence_id
                    parsed.append(measurement)

                evidence.append(unit)

    validate_measurements(parsed)
    return EvidenceStore(document.document_id, evidence)
