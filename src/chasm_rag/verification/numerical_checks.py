from __future__ import annotations

import re

from ..evidence.evidence_models import EvidenceUnit


def numbers(text: str) -> set[str]:
    return set(re.findall(r"[-+]?\d+(?:\.\d+)?", text))


def exact_number_support(claim: str, evidence: list[EvidenceUnit]) -> tuple[bool, str | None]:
    claimed = numbers(claim)
    if not claimed:
        return True, None
    source_numbers = set().union(*(numbers(item.original_text) for item in evidence))
    if claimed.issubset(source_numbers):
        return True, None
    # A report-grounded calculation is allowed only for the explicit lipid
    # identity used here; it remains a derived fact, never a source quotation.
    if ("subtracting hdl" in claim.casefold() or "derived arithmetic" in claim.casefold() or "derived check" in claim.casefold()) and "total cholesterol" in claim.casefold():
        measurements = [item.measurement for item in evidence if item.measurement and item.measurement.value is not None]
        values = {item.test_name.casefold(): item.value for item in measurements}
        total = values.get("cholesterol total", values.get("total cholesterol"))
        hdl = values.get("hdl cholesterol")
        if total is not None and hdl is not None:
            derived = f"{total - hdl:g}"
            if claimed.difference(source_numbers).issubset({derived}):
                return True, None
    return False, "Claimed numerical value is absent from retrieved source evidence."


def exact_unit_support(claim: str, evidence: list[EvidenceUnit]) -> tuple[bool, str | None]:
    claimed_units = set(re.findall(r"\b(?:mg/dL|mmol/L|g/dL|mEq/L|%|IU/L)\b", claim, re.I))
    if not claimed_units:
        return True, None
    source_units = set().union(*(set(re.findall(r"\b(?:mg/dL|mmol/L|g/dL|mEq/L|%|IU/L)\b", item.original_text, re.I)) for item in evidence))
    return (claimed_units.issubset(source_units), None if claimed_units.issubset(source_units) else "Claimed unit is absent from retrieved source evidence.")

