from __future__ import annotations

import re

from ..evidence.evidence_models import EvidenceUnit, Measurement
from ..evidence.measurement_validator import classify_measurement, is_elevated


_LIPID_NAMES = ("cholesterol total", "total cholesterol", "triglycerides", "hdl cholesterol", "ldl cholesterol", "vldl cholesterol", "non-hdl cholesterol")


def is_lipid_panel(measurements: list[Measurement]) -> bool:
    names = {item.test_name.casefold() for item in measurements}
    return len(names.intersection(_LIPID_NAMES)) >= 3


def _display_name(item: Measurement) -> str:
    return "Total cholesterol" if item.test_name.casefold() == "cholesterol total" else item.test_name


def _threshold(item: Measurement) -> str:
    reference = item.reference_range
    if not reference or reference.threshold is None:
        return reference.raw if reference and reference.raw else "the supplied range"
    return f"{reference.threshold:g} {item.unit or ''}".strip()


def _normalized_test_name(item: Measurement) -> str:
    return re.sub(r"[^a-z0-9]", "", item.test_name.casefold())


def _reported_number(item: Measurement) -> str:
    if item.value is None:
        return "not stated"
    source_values = re.findall(r"[-+]?\d+(?:\.\d+)?", item.original_text)
    return next(
        (token for token in source_values if float(token) == item.value),
        f"{item.value:g}",
    )


def _find_thyroid_measurement(
    measurements: list[Measurement], names: set[str]
) -> Measurement | None:
    matches = [item for item in measurements if _normalized_test_name(item) in names]
    return matches[0] if len(matches) == 1 else None


def _measurement_description(item: Measurement, name: str) -> str:
    value = "not reported" if item.value is None else f"{item.value:g} {item.unit or ''}".strip()
    reference = (
        f" (reference range: {item.reference_range.raw})"
        if item.reference_range and item.reference_range.raw
        else ""
    )
    status = classify_measurement(item)
    if status.startswith("HIGH"):
        comparison = "above"
    elif status.startswith("LOW"):
        comparison = "below"
    elif status == "WITHIN SUPPLIED RANGE":
        comparison = "within"
    else:
        comparison = "not assessed because no reference range was supplied"
    page = f" [p. {item.page}]"
    if status == "NOT ASSESSED":
        return f"{name} at {value} was {comparison}{page}"
    return f"{name} at {value}{reference} is {comparison} the supplied reference range{page}"


def conservative_thyroid_summary(
    measurements: list[Measurement], evidence: list[EvidenceUnit]
) -> str | None:
    """Summarize a complete, unambiguous T3/T4/TSH pattern from report evidence."""
    t3 = _find_thyroid_measurement(
        measurements,
        {"t3", "t3total", "totalt3", "triiodothyronine", "triiodothyroninetotal", "totaltriiodothyronine"},
    )
    t4 = _find_thyroid_measurement(
        measurements,
        {"t4", "t4total", "totalt4", "thyroxine", "thyroxinetotal", "totalthyroxine"},
    )
    tsh = _find_thyroid_measurement(
        measurements,
        {"tsh", "thyroidstimulatinghormone"},
    )
    if t3 is None or t4 is None or tsh is None:
        return None

    lines = [
        "The thyroid profile reports "
        + "; ".join(
            (
                _measurement_description(t3, "total T3"),
                _measurement_description(t4, "total T4"),
                _measurement_description(tsh, "TSH"),
            )
        )
        + "."
    ]
    if (
        is_elevated(t3)
        and is_elevated(t4)
        and classify_measurement(tsh).startswith("LOW")
        and not (t3.consistency_warnings or t4.consistency_warnings or tsh.consistency_warnings)
    ):
        lines.append(
            "Together, this is a biochemical pattern commonly associated with "
            "hyperthyroidism, but it is not by itself a confirmed clinical diagnosis."
        )

    source_text = " ".join(item.original_text for item in evidence).casefold()
    if re.search(r"\bcircadian variation\b|\bvar(?:y|ies|iation)\b.{0,40}\bday\b", source_text):
        lines.append("The report notes that TSH levels vary over the course of the day.")
    if "binding protein" in source_text and (
        "pregnancy" in source_text or "steroid therapy" in source_text
    ):
        lines.append(
            "It also notes that changes in thyroid-hormone-binding proteins, "
            "particularly during pregnancy or steroid therapy, may affect total T3 and T4."
        )
    if (
        re.search(r"\bfree\s*[, ]?\s*t[34]\b", source_text)
        and re.search(r"\bcorrelat\w*\s+more closely\b", source_text)
    ):
        lines.append(
            "The report states that free T3 and free T4 may correlate more closely "
            "with clinical status than total hormone measurements."
        )
    return " ".join(lines)


def conservative_measurement_summary(measurements: list[Measurement]) -> str:
    if not measurements:
        return "No structured measurements were available for a source-derived summary."
    counts = {
        "high": sum(classify_measurement(item).startswith("HIGH") for item in measurements),
        "low": sum(classify_measurement(item).startswith("LOW") for item in measurements),
        "within": sum(
            classify_measurement(item) == "WITHIN SUPPLIED RANGE"
            for item in measurements
        ),
        "unassessed": sum(
            classify_measurement(item) == "NOT ASSESSED" for item in measurements
        ),
    }
    overview = (
        f"The report contains {len(measurements)} structured measurements: "
        f"{counts['high']} above, {counts['low']} below, "
        f"{counts['within']} within their supplied reference ranges, and "
        f"{counts['unassessed']} not assessable from a supplied range."
    )
    rows = [
        f"- {item.test_name}: reported value "
        f"{_reported_number(item)}"
        f"{f' {item.unit}' if item.unit else ''}; supplied reference range "
        f"{item.reference_range.raw if item.reference_range and item.reference_range.raw else 'not stated'}; "
        f"calculated classification: {classify_measurement(item)}; "
        f"reported flag: {item.reported_flag or 'Not stated'} [p. {item.page}]."
        for item in measurements
    ]
    warnings = [
        f"- {item.test_name}: {warning}"
        for item in measurements
        for warning in item.consistency_warnings
    ]
    result = "\n".join([overview, *rows])
    if warnings:
        result += "\nSource/value consistency checks (reported values are unchanged):\n"
        result += "\n".join(warnings)
    return result


def conservative_lipid_summary(measurements: list[Measurement], report_instructions: list[str]) -> str:
    """Create a source-derived lipid summary with all arithmetic explicitly labeled."""
    sentences = [
        "The lipid profile reports the following structured measurements:",
        conservative_measurement_summary(measurements),
    ]
    if report_instructions:
        sentences.append(f"The report notes: {report_instructions[0]}")
    if any(
        classify_measurement(item).startswith(("HIGH", "LOW"))
        or (
            item.reported_flag
            and item.reported_flag.strip().upper()
            in {"H", "HIGH", "L", "LOW", "ABNORMAL", "BORDERLINE"}
        )
        for item in measurements
    ):
        sentences.append(
            "These source-derived classifications are not a clinical diagnosis; "
            "review the report with a qualified healthcare professional."
        )
    return " ".join(sentences)
