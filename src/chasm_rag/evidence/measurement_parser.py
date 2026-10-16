from __future__ import annotations

import re

from .evidence_models import Measurement, ReferenceRange

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_RANGE = re.compile(rf"(?P<low>{_NUMBER})\s*(?:-|to)\s*(?P<high>{_NUMBER})")
_THRESHOLD = re.compile(rf"(?P<op><=|>=|<|>)\s*(?P<value>{_NUMBER})")
_MEASUREMENT = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9()/% .,'-]{{1,80}}?)\s*(?::|\||\t|\s{{2,}}|\s+(?=(?:{_NUMBER}|N/?A|Not\s+reported|--)(?:\s|$)))\s*(?P<value>{_NUMBER}|N/?A|Not\s+reported|--)?\s*(?P<unit>[A-Za-zµμ/%][A-Za-z0-9µμ/%^.-]*|mmol/L|mg/dL)?\s*(?P<tail>.*)$",
    re.IGNORECASE,
)
_SOURCE_FLAG = re.compile(
    r"\b(HIGH|LOW|ABNORMAL|NORMAL|H|L|BORDERLINE)\b",
    re.IGNORECASE,
)
_SOURCE_FLAGS = {"high", "low", "abnormal", "normal", "h", "l", "borderline"}
_METADATA_LABELS = {
    "registered on", "collected on", "reported on", "generated on", "age", "sex", "pid",
    "sample collected at", "sample type", "ref. by", "reference by", "patient id", "uhid", "name",
    "tat", "to check report authenticity",
}
_UNIT_WITHOUT_RANGE = {
    "cell",
    "cells",
    "count",
    "cumm",
    "fl",
    "iu",
    "kg",
    "l",
    "mg",
    "ml",
    "mm",
    "mmhg",
    "pg",
    "sec",
    "u",
    "unit",
    "units",
}
_LIPID_TESTS = (
    "non-hdl cholesterol", "vldl cholesterol", "ldl cholesterol", "hdl cholesterol",
    "cholesterol total", "total cholesterol", "triglycerides",
)


def _range(tail: str) -> ReferenceRange | None:
    match = _RANGE.search(tail)
    if match:
        return ReferenceRange(float(match["low"]), float(match["high"]), raw=match.group())
    match = _THRESHOLD.search(tail)
    if match:
        return ReferenceRange(comparison=match["op"], threshold=float(match["value"]), raw=match.group())
    return None


def _unit_after_range(tail: str) -> str | None:
    reference = _RANGE.search(tail) or _THRESHOLD.search(tail)
    remaining = tail[reference.end():] if reference else tail
    for token in re.findall(r"[A-Za-zµμ%][A-Za-z0-9µμ/%^.-]*", remaining):
        if token.casefold() not in _SOURCE_FLAGS | {"normal", "reference", "range"}:
            return token
    return None


def _source_flag(text: str | None) -> str | None:
    if not text:
        return None
    match = _SOURCE_FLAG.fullmatch(text.strip())
    if not match:
        return None
    return match.group(1)


def reported_flag_in_text(text: str | None) -> str | None:
    if not text:
        return None
    match = _SOURCE_FLAG.search(text)
    return _source_flag(match.group(1)) if match else None


def parse_measurement_fields(
    name: str,
    value: str,
    reference: str,
    unit: str | None,
    reported_flag: str | None,
    page: int,
    original_text: str,
) -> Measurement | None:
    """Build a measurement from separately extracted table cells."""
    if not re.fullmatch(_NUMBER, value.strip()):
        return None
    parsed_range = _range(reference)
    if not parsed_range:
        return None
    flag = reported_flag_in_text(reported_flag)
    parsed_unit = unit.strip() if unit and unit.strip() else None
    unit_flag = _source_flag(parsed_unit)
    if unit_flag:
        flag = flag or unit_flag
        parsed_unit = None
    if not name.strip():
        return None
    return Measurement(
        re.sub(r"\s+", " ", name).strip(),
        float(value),
        parsed_unit or None,
        parsed_range,
        flag,
        page,
        original_text,
    )


def parse_measurement_line(line: str, page: int) -> Measurement | None:
    """Extract only explicit report-shaped lines; narrative numbers remain narrative evidence."""
    match = _MEASUREMENT.match(line.strip())
    if not match or not match["value"]:
        return None
    value_text = match["value"]
    value = float(value_text) if re.fullmatch(_NUMBER, value_text) else None
    tail = match["tail"]
    ref = _range(tail)
    unit = match["unit"]
    flag = reported_flag_in_text(f"{unit or ''} {tail}")
    if (
        not unit
        or _source_flag(unit)
        or unit.casefold() in {"ref", "reference", "range"}
    ):
        unit = _unit_after_range(tail)
    name = re.sub(r"\s+", " ", match["name"]).strip(" |:")
    if (len(name) < 2 or any(name.casefold().startswith(label) for label in _METADATA_LABELS) or
            (value is None and not re.search(r"N/?A|Not\s+reported|--", value_text, re.I))):
        return None
    # Dates/identifiers often have the same "label: number" surface form as a
    # measurement. Without a unit, range, or explicit flag they are not evidence
    # of a laboratory result.
    if (
        ref is None
        and flag is None
        and (
            not unit
            or (
                "/" not in unit
                and "%" not in unit
                and unit.casefold() not in _UNIT_WITHOUT_RANGE
            )
        )
    ):
        return None
    return Measurement(name, value, unit, ref, flag, page, line, "parsed" if value is not None else "missing")


def parse_measurements(text: str, page: int) -> list[Measurement]:
    return [result for line in text.splitlines() if (result := parse_measurement_line(line, page))]


_NON_MEASUREMENT_LABELS = {
    "blood indices",
    "complete blood count",
    "differential wbc count",
    "hemoglobin",
    "instrument",
    "instruments",
    "interpretation",
    "lipid profile",
    "note",
    "notes",
    "primary sample type",
    "rbc count",
    "sample type",
    "wbc count",
}
_METHOD_LABELS = {
    "calculated",
    "electrical impedance vcs",
    "eia",
    "ifcc amp",
    "immunoturbidimetry",
    "indirect ise",
    "molybdate uv",
    "modified jaffe kinetic",
    "spectrophotometry",
    "urease uv",
    "uricase",
}


def _fragment_test_name(line: str) -> str | None:
    first_cell = line.split("|", 1)[0].strip()
    name = re.split(
        r"\s+(?=(?:[-+]?\d|<=|>=|<|>|HIGH\b|LOW\b|NORMAL\b|ABNORMAL\b|"
        r"BORDERLINE\b|H\b|L\b))",
        first_cell,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" :-|")
    normalized = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    if (
        not normalized
        or not re.search(r"[a-z]", normalized)
        or _source_flag(name)
        or normalized in _NON_MEASUREMENT_LABELS | _METHOD_LABELS
    ):
        return None
    if len(name) > 5 and name == name.upper():
        return None
    if re.search(r"\b(?:recommendations?|optimal|very high|above optimal)\b", name, re.I):
        return None
    if re.search(r"\b(?:note|interpretation|advice|guideline)\b", name, re.I):
        return None
    return name


def parse_fragmented_measurements(text: str, page: int) -> list[Measurement]:
    """Reconstruct report rows whose analyte and result fields occupy nearby lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results: list[Measurement] = []
    seen: set[tuple[str, float | None, str | None]] = set()

    for start, line in enumerate(lines):
        name = _fragment_test_name(line)
        if name is None:
            continue

        grouped = [line]
        reference_match = _RANGE.search(line) or _THRESHOLD.search(line)
        end = start
        def has_reported_value(values: list[str]) -> bool:
            combined_values = "\n".join(values)
            if reference_match is None:
                return False
            without_reference = (
                combined_values[:reference_match.start()]
                + combined_values[reference_match.end():]
            )
            without_name = re.sub(
                re.escape(name), "", without_reference, count=1, flags=re.I
            )
            return bool(
                re.findall(
                    _NUMBER,
                    _SOURCE_FLAG.sub("", without_name),
                )
            )

        for index in range(start + 1, min(len(lines), start + 5)):
            next_line = lines[index]
            if has_reported_value(grouped):
                break
            if _fragment_test_name(next_line):
                break
            grouped.append(next_line)
            end = index
            reference_match = _RANGE.search("\n".join(grouped)) or _THRESHOLD.search(
                "\n".join(grouped)
            )

        combined = "\n".join(grouped)
        reference_match = _RANGE.search(combined) or _THRESHOLD.search(combined)
        if reference_match is None:
            continue

        without_reference = (
            combined[:reference_match.start()]
            + combined[reference_match.end():]
        )
        without_name = re.sub(
            re.escape(name), "", without_reference, count=1, flags=re.I
        )
        value_tokens = re.findall(
            _NUMBER,
            _SOURCE_FLAG.sub("", without_name),
        )
        if not value_tokens:
            continue

        value = float(value_tokens[0])
        reference = ReferenceRange(
            float(reference_match["low"]) if "low" in reference_match.groupdict() and reference_match["low"] else None,
            float(reference_match["high"]) if "high" in reference_match.groupdict() and reference_match["high"] else None,
            reference_match["op"] if "op" in reference_match.groupdict() else None,
            float(reference_match["value"]) if "value" in reference_match.groupdict() and reference_match["value"] else None,
            reference_match.group(),
        )
        unit = _unit_after_range(combined[reference_match.end():])
        flag = reported_flag_in_text(combined)
        signature = (name.casefold(), value, unit)
        if signature in seen:
            continue
        seen.add(signature)
        results.append(
            Measurement(
                re.sub(r"\s+", " ", name).strip(),
                value,
                unit,
                reference,
                flag,
                page,
                "\n".join(lines[start:end + 1]),
            )
        )

    return results


def _first_unit(lines: list[str]) -> str | None:
    for line in lines:
        candidate = line.strip()
        # Restrict this to actual measurement-unit shapes. Technique labels such
        # as "Spectrophotometry" and "Calculated" are never units.
        if (candidate == "%" or "/" in candidate or
                candidate.casefold() in {"meq/l", "mmol/l", "iu/l", "u/l", "g/l", "mg/l", "ng/ml", "pg/ml"}):
            return candidate
    return None


def _first_range(lines: list[str]) -> ReferenceRange | None:
    return next((found for line in lines if (found := _range(line))), None)


def _last_range(lines: list[str]) -> ReferenceRange | None:
    return next((found for line in reversed(lines) if (found := _range(line))), None)


def _standalone_values(lines: list[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        clean = line.strip()
        if re.fullmatch(_NUMBER, clean):
            values.append(float(clean))
    return values


def _flag_near_reference(lines: list[str], prefer_last: bool = False) -> str | None:
    indexes = [index for index, line in enumerate(lines) if _range(line) is not None]
    if prefer_last:
        indexes = indexes[-1:]
    for index in indexes:
        line = lines[index]
        if _range(line) is None:
            continue
        nearby = lines[max(0, index - 1):index + 3]
        if any(re.fullmatch(r"(?:high|h)", value, re.I) for value in nearby):
            return "HIGH"
        if any(re.fullmatch(r"(?:low|l)", value, re.I) for value in nearby):
            return "LOW"
    return None


def parse_lipid_profile_layout(text: str, page: int) -> list[Measurement]:
    """Reconstruct common column-major lipid PDF text without treating demographics as tests.

    Some PDFs expose a visual row as separate text columns (unit/range/flag/value/test),
    sometimes in reverse order. This parser uses the report's LIPID PROFILE section and
    named analytes only; it does not guess values from demographics or dates.
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    try:
        start = next(index for index, line in enumerate(lines) if line.casefold() == "lipid profile") + 1
    except StopIteration:
        return []
    end = next((index for index in range(start, len(lines)) if "smart pathology laboratory" in lines[index].casefold() or lines[index].casefold().endswith("recommendations")), len(lines))
    section = lines[start:end]
    positions: list[tuple[int, str]] = []
    for index, line in enumerate(section):
        normalized = line.casefold()
        test = next((candidate for candidate in _LIPID_TESTS if normalized == candidate), None)
        if test:
            positions.append((index, line))
    results: list[Measurement] = []
    for index, (position, name) in enumerate(positions):
        left_boundary = positions[index - 1][0] + 1 if index else 0
        right_boundary = positions[index + 1][0] if index + 1 < len(positions) else len(section)
        before, after = section[left_boundary:position], section[position + 1:right_boundary]
        before_reference, after_reference = _last_range(before), _first_range(after)
        # A conventional row exposed in reading order starts immediately with
        # the unit after the analyte name. Otherwise, this PDF family prints
        # the columns before the analyte name, so use the preceding side.
        conventional_after_row = bool(after and _first_unit(after[:1]) and after_reference)
        source_side = after if conventional_after_row else before
        other_side = before if conventional_after_row else after
        reference = (_first_range(source_side) if conventional_after_row else _last_range(source_side)) or _first_range(other_side)
        unit = _first_unit(source_side) or _first_unit(other_side)
        after_values, before_values = _standalone_values(after), _standalone_values(before)
        value = after_values[0] if after_values else (before_values[-1] if before_values else None)
        if value is None or unit is None or reference is None:
            continue
        flag = _flag_near_reference(source_side, prefer_last=not conventional_after_row)
        original = f"{name} | {value:g} {unit} | Reference {reference.raw}" + (" | High" if flag else "")
        results.append(Measurement(name, value, unit, reference, flag, page, original))
    return results
