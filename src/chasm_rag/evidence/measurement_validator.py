from __future__ import annotations

from collections import defaultdict

from .evidence_models import Measurement


def classify_measurement(item: Measurement) -> str:
    """Classify the numeric result against its supplied range, not its source flag."""
    if item.value is None or item.reference_range is None:
        return "NOT ASSESSED"
    reference = item.reference_range
    if reference.upper is not None and item.value > reference.upper:
        return "HIGH (above supplied range)"
    if reference.lower is not None and item.value < reference.lower:
        return "LOW (below supplied range)"
    if (
        reference.comparison == "<"
        and reference.threshold is not None
        and item.value >= reference.threshold
    ) or (
        reference.comparison == "<="
        and reference.threshold is not None
        and item.value > reference.threshold
    ):
        return "HIGH (above supplied threshold)"
    if (
        reference.comparison == ">"
        and reference.threshold is not None
        and item.value <= reference.threshold
    ) or (
        reference.comparison == ">="
        and reference.threshold is not None
        and item.value < reference.threshold
    ):
        return "LOW (below supplied threshold)"
    return "WITHIN SUPPLIED RANGE"


def is_elevated(item: Measurement) -> bool:
    return classify_measurement(item).startswith("HIGH")


def validate_measurements(measurements: list[Measurement]) -> list[Measurement]:
    """Flags arithmetic/report inconsistencies; it never changes a reported value."""
    seen: dict[tuple[str, str | None], list[Measurement]] = defaultdict(list)
    for item in measurements:
        if item.value is not None and item.reference_range:
            ref = item.reference_range
            out_of_range = (
                (ref.lower is not None and item.value < ref.lower)
                or (ref.upper is not None and item.value > ref.upper)
                or (
                    ref.comparison in {"<", "<="}
                    and ref.threshold is not None
                    and (
                        item.value >= ref.threshold
                        if ref.comparison == "<"
                        else item.value > ref.threshold
                    )
                )
                or (
                    ref.comparison in {">", ">="}
                    and ref.threshold is not None
                    and (
                        item.value <= ref.threshold
                        if ref.comparison == ">"
                        else item.value < ref.threshold
                    )
                )
            )
            flag = item.reported_flag.strip().upper() if item.reported_flag else ""
            if flag in {"H", "HIGH", "L", "LOW", "ABNORMAL"} and not out_of_range:
                item.consistency_warnings.append("Source abnormal flag conflicts with parsed reference range.")
            elif flag == "NORMAL" and out_of_range:
                item.consistency_warnings.append(
                    "Source normal flag conflicts with the value's calculated range classification."
                )
            elif flag == "BORDERLINE" and out_of_range:
                item.consistency_warnings.append(
                    "Source borderline flag differs from the value's calculated out-of-range classification."
                )
        seen[(item.test_name.casefold(), item.unit)].append(item)
    for duplicates in seen.values():
        values = {item.value for item in duplicates if item.value is not None}
        if len(values) > 1:
            for item in duplicates:
                item.consistency_warnings.append("Repeated measurement has conflicting reported values; preserve both source values.")
    _validate_lipid_arithmetic(measurements)
    return measurements


def _validate_lipid_arithmetic(measurements: list[Measurement]) -> None:
    """Label a transparent derived check; never change the report's stated value."""
    by_name = {item.test_name.casefold(): item for item in measurements if item.value is not None}
    total = by_name.get("cholesterol total") or by_name.get("total cholesterol")
    hdl = by_name.get("hdl cholesterol")
    non_hdl = by_name.get("non-hdl cholesterol")
    if not total or not hdl or not non_hdl or not total.unit or total.unit != hdl.unit or total.unit != non_hdl.unit:
        return
    derived = total.value - hdl.value
    if abs(derived - non_hdl.value) > 1e-6:
        non_hdl.consistency_warnings.append(
            f"Derived check from report values: total cholesterol ({total.value:g}) - HDL cholesterol ({hdl.value:g}) = {derived:g} {total.unit}, which differs from the reported non-HDL value ({non_hdl.value:g} {non_hdl.unit})."
        )
