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
        item.reference_range.raw
        if item.reference_range and item.reference_range.raw
        else None
    )
    status = classify_measurement(item)
    if status.startswith("HIGH"):
        comparison = "above"
    elif status.startswith("LOW"):
        comparison = "below"
    elif status == "WITHIN SUPPLIED RANGE":
        comparison = "within"
    else:
        comparison = None

    if status == "NOT ASSESSED":
        return (
            f"{name} is {value}. The report does not provide a reference "
            f"range (page {item.page})."
        )

    range_text = (
        f" of {reference}"
        if reference
        else ""
    )
    if comparison == "within":
        return (
            f"{name} is {value}, within the supplied reference range{range_text} "
            f"(page {item.page})."
        )
    return (
        f"{name} is {value}, which is {comparison} the supplied "
        f"reference range{range_text} (page {item.page})."
    )


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

    paragraphs = [
        "**Reported thyroid results**\n\n"
        + "\n".join(
            (
                f"- {_measurement_description(t3, 'Total T3')}",
                f"- {_measurement_description(t4, 'Total T4')}",
                f"- {_measurement_description(tsh, 'TSH')}",
            )
        )
    ]
    if (
        is_elevated(t3)
        and is_elevated(t4)
        and classify_measurement(tsh).startswith("LOW")
        and not (t3.consistency_warnings or t4.consistency_warnings or tsh.consistency_warnings)
    ):
        paragraphs.append(
            "Together, this is a biochemical pattern commonly associated with "
            "hyperthyroidism, but it is not by itself a confirmed clinical diagnosis."
        )

    source_text = " ".join(item.original_text for item in evidence).casefold()
    if re.search(r"\bcircadian variation\b|\bvar(?:y|ies|iation)\b.{0,40}\bday\b", source_text):
        paragraphs.append("The report notes that TSH levels vary over the course of the day.")
    if "binding protein" in source_text and (
        "pregnancy" in source_text or "steroid therapy" in source_text
    ):
        paragraphs.append(
            "It also notes that changes in thyroid-hormone-binding proteins, "
            "particularly during pregnancy or steroid therapy, may affect total T3 and T4."
        )
    if (
        re.search(r"\bfree\s*[, ]?\s*t[34]\b", source_text)
        and re.search(r"\bcorrelat\w*\s+more closely\b", source_text)
    ):
        paragraphs.append(
            "The report states that free T3 and free T4 may correlate more closely "
            "with clinical status than total hormone measurements."
        )
    return "\n\n".join(paragraphs)


def conservative_measurement_summary(measurements: list[Measurement]) -> str:
    if not measurements:
        return "No structured measurements were available for a source-derived summary."
    within = [
        item
        for item in measurements
        if classify_measurement(item) == "WITHIN SUPPLIED RANGE"
    ]
    outside = [
        item
        for item in measurements
        if classify_measurement(item).startswith(("HIGH", "LOW"))
    ]
    unassessed = [
        item for item in measurements if classify_measurement(item) == "NOT ASSESSED"
    ]

    def describe(item: Measurement) -> str:
        value = _reported_number(item)
        if item.unit:
            value += f" {item.unit}"
        reference = (
            item.reference_range.raw.strip()
            if item.reference_range and item.reference_range.raw
            else None
        )
        if reference:
            reference = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", reference)

        classification = classify_measurement(item)
        label = _display_name(item)
        if classification == "WITHIN SUPPLIED RANGE":
            result = f"The report gives {label} as {value}, which is within its reference range"
            if reference:
                result += f" of {reference}"
        elif classification.startswith("HIGH"):
            result = f"The report gives {label} as {value}, which is above its reference range"
            if reference:
                result += f" of {reference}"
        elif classification.startswith("LOW"):
            result = f"The report gives {label} as {value}, which is below its reference range"
            if reference:
                result += f" of {reference}"
        else:
            if item.value is None:
                result = f"The report does not list a result for {label}"
            else:
                result = f"The report gives {label} as {value}"
            result += " and does not provide a reference range"
        return f"{result} (page {item.page})."

    sections = [f"The report lists {len(measurements)} measurements."]
    if within:
        sections.extend(
            [
                "",
                "**Within the supplied reference ranges**",
                *[f"- {describe(item)}" for item in within],
            ]
        )
    if outside:
        sections.extend(
            [
                "",
                "**Outside the supplied reference ranges**",
                *[f"- {describe(item)}" for item in outside],
            ]
        )
    if unassessed:
        sections.extend(
            [
                "",
                "**Reference range not available**",
                *[f"- {describe(item)}" for item in unassessed],
            ]
        )

    notes = []
    for item in measurements:
        for warning in item.consistency_warnings:
            if "Source normal flag conflicts" in warning:
                direction = (
                    "below"
                    if classify_measurement(item).startswith("LOW")
                    else "above"
                )
                notes.append(
                    f"{_display_name(item)} is {direction} its supplied range "
                    "even though the report marks it as normal. Please check this "
                    f"discrepancy against the original report (page {item.page})."
                )
            elif "Source abnormal flag conflicts" in warning:
                notes.append(
                    f"The report marks {_display_name(item)} as abnormal, although "
                    "its value falls within the supplied range. Please check the "
                    f"original report (page {item.page})."
                )
            elif "Source borderline flag differs" in warning:
                notes.append(
                    f"The report's borderline flag for {_display_name(item)} differs "
                    "from the comparison with its supplied range. Please check the "
                    f"original report (page {item.page})."
                )
            elif "Repeated measurement has conflicting reported values" in warning:
                notes.append(
                    f"The report contains different values for {_display_name(item)}. "
                    f"Both have been retained for review (page {item.page})."
                )
            elif warning.startswith("Derived check from report values:"):
                if "non-hdl" in item.test_name.casefold():
                    notes.append(
                        "A calculation using total cholesterol and HDL cholesterol "
                        "gives a different result from the non-HDL cholesterol value "
                        "printed in the report. The reported value has been kept "
                        "unchanged."
                    )
                else:
                    notes.append(
                        f"A calculation using other reported values differs from "
                        f"{_display_name(item)} in the report. The reported value "
                        "has been kept unchanged."
                    )
            else:
                notes.append(
                    f"Please review the source note for {_display_name(item)} "
                    f"(page {item.page})."
                )
    if notes:
        sections.extend(
            [
                "",
                "**Please review these report inconsistencies**",
                *[f"- {note}" for note in dict.fromkeys(notes)],
            ]
        )
    return "\n".join(sections)


def conservative_lipid_summary(measurements: list[Measurement], report_instructions: list[str]) -> str:
    """Create a source-derived lipid summary with all arithmetic explicitly labeled."""
    sentences = [
        "The lipid profile reports these results:",
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
    return "\n\n".join(sentences)
