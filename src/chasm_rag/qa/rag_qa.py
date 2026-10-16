from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from ..evidence.evidence_models import EvidenceUnit, Measurement, VerificationStatus
from ..evidence.measurement_validator import classify_measurement
from ..retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from ..verification.claim_verifier import ClaimVerification


class AnswerGenerator(Protocol):
    """Optional local or application-provided question-and-evidence answerer."""

    def generate(
        self,
        question: str,
        evidence: list[EvidenceUnit],
        feedback: str | None = None,
    ) -> str: ...


class AnswerVerifier(Protocol):
    def verify(
        self,
        text: str,
        candidate_evidence: list[EvidenceUnit] | None = None,
    ) -> list[ClaimVerification]: ...


@dataclass
class GroundedAnswer:
    answer: str
    evidence: list[RetrievalResult]
    verification: list[ClaimVerification]
    uncertainty: str | None = None


_BLOCKING_STATUSES = {
    VerificationStatus.CONTRADICTED,
    VerificationStatus.UNSUPPORTED,
}
_ABNORMAL_STATUSES = ("HIGH", "LOW")
_LIPID_TERMS = ("cholesterol", "triglyceride", "hdl", "ldl", "vldl", "lipid")


class RAGQuestionAnswerer:
    """Answers from one selected document and verifies against its cited evidence."""

    def __init__(
        self,
        retriever: HybridRetriever,
        verifier: AnswerVerifier,
        answer_generator: AnswerGenerator | None = None,
        selected_document_id: str | None = None,
        valid_pages: set[int] | None = None,
        selected_document_name: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.verifier = verifier
        self.answer_generator = answer_generator
        self.valid_pages = valid_pages
        self.selected_document_name = selected_document_name
        document_ids = {item.document_id for item in retriever.evidence if item.document_id}
        if selected_document_id is None and len(document_ids) == 1:
            selected_document_id = next(iter(document_ids))
        self.selected_document_id = selected_document_id
        self._source_evidence = [
            item for item in retriever.evidence
            if selected_document_id is not None
            and item.document_id == selected_document_id
        ]

    
    
    
    def answer(self, question: str) -> GroundedAnswer:
        retrieved, invalid_retrieved = self._retrieved_for_selected_document(question)
        all_measurements = [
            item for item in self._source_evidence if item.measurement is not None
        ]
        structured = self._structured_selection(question, all_measurements)
        if structured is not None:
            units, answer, invalid_structured = structured
            return self._apply_verification_policy(
                question,
                answer,
                units,
                retrieved,
                allow_retry=False,
                citation_issues=invalid_retrieved + invalid_structured,
            )
        if not all_measurements and _is_measurement_question(question, []):
            citation_note = (
                " Relevant retrieved chunks also have missing or invalid citation metadata."
                if invalid_retrieved
                else ""
            )
            return GroundedAnswer(
                "The selected document contains no structured measurements to classify against reference ranges.",
                [],
                [],
                "Measurement classification was unavailable because no structured measurements were extracted."
                + citation_note,
            )

        evidence = [item.evidence for item in retrieved]
        if not evidence:
            return GroundedAnswer(
                "The selected document does not contain usable evidence to answer this question.",
                [],
                [],
                self._no_evidence_reason(invalid_retrieved),
            )
        if self.answer_generator is None:
            candidate = _extractive_answer(evidence)
        else:
            candidate = self.answer_generator.generate(question, evidence)
        return self._apply_verification_policy(
            question,
            candidate,
            evidence,
            retrieved,
            allow_retry=True,
            citation_issues=invalid_retrieved,
        )

    def _retrieved_for_selected_document(
        self, question: str
    ) -> tuple[list[RetrievalResult], int]:
        if not self._source_evidence:
            return [], 0
        allowed = {id(item) for item in self._source_evidence}
        selected = [
            result for result in self.retriever.search(question)
            if id(result.evidence) in allowed
        ]
        valid = [
            result for result in selected
            if _has_valid_citation(
                result.evidence, self.selected_document_id, self.valid_pages
            )
        ]
        return valid, len(selected) - len(valid)

    def _structured_selection(
        self,
        question: str,
        measurement_units: list[EvidenceUnit],
    ) -> tuple[list[EvidenceUnit], str, int] | None:
        if not measurement_units or not _is_measurement_question(
            question, [item.measurement for item in measurement_units if item.measurement]
        ):
            return None

        measurements = [
            item.measurement for item in measurement_units if item.measurement
        ]
        selected: list[Measurement]
        if _asks_within_range(question):
            selected = [
                item for item in measurements
                if classify_measurement(item) == "WITHIN SUPPLIED RANGE"
            ]
            if not selected:
                return (
                    [],
                    "No extracted measurement is classified within its supplied reference range.",
                    0,
                )
        elif _asks_for_abnormal_measurements(question):
            selected = [
                item for item in measurements
                if classify_measurement(item).startswith(_ABNORMAL_STATUSES)
                or classify_measurement(item) == "NOT ASSESSED"
            ]
            if not selected:
                selected = measurements
        elif _asks_for_all_measurements(question):
            selected = measurements
        else:
            selected = _mentioned_measurements(question, measurements)
            if not selected and _mentions_lipid_profile(question):
                selected = [
                    item for item in measurements if _is_lipid_measurement(item.test_name)
                ]
        if not selected:
            return None

        selected_ids = {id(item) for item in selected}
        selected_units = [
            unit for unit in measurement_units
            if unit.measurement is not None and id(unit.measurement) in selected_ids
        ]
        cited_units = [
            unit for unit in selected_units
            if _has_valid_citation(unit, self.selected_document_id, self.valid_pages)
        ]
        answer = _measurement_answer(selected, question)
        return cited_units, answer, len(selected_units) - len(cited_units)

    def _apply_verification_policy(
        self,
        question: str,
        candidate: str,
        evidence: list[EvidenceUnit],
        retrieved: list[RetrievalResult],
        allow_retry: bool,
        citation_issues: int = 0,
    ) -> GroundedAnswer:
        verification = self.verifier.verify(candidate, evidence)
        generation_failed = not candidate.strip() or _has_blocking_claims(verification)
        if generation_failed and allow_retry and self.answer_generator is not None:
            feedback = _verification_feedback(verification)
            candidate = self.answer_generator.generate(question, evidence, feedback)
            verification = self.verifier.verify(candidate, evidence)
            generation_failed = not candidate.strip() or _has_blocking_claims(verification)

        uncertainty_parts: list[str] = []
        if generation_failed:
            candidate = _extractive_answer(evidence)
            verification = self.verifier.verify(candidate, evidence)
            if not candidate.strip() or _has_blocking_claims(verification):
                citation_note = (
                    f" Citation metadata is missing or invalid for "
                    f"{citation_issues} selected evidence unit(s)."
                    if citation_issues
                    else ""
                )
                return GroundedAnswer(
                    "The answer could not be verified against the selected document's source evidence.",
                    retrieved,
                    verification,
                    "Both the generated answer and conservative source excerpts failed "
                    f"verification.{citation_note}",
                )
            uncertainty_parts.append(
                "The generated answer was not supported; a conservative extractive answer is shown instead."
            )

        if any(
            item.status in {
                VerificationStatus.UNCERTAIN,
                VerificationStatus.SOURCE_INCONSISTENT,
            }
            for item in verification
        ) or any(
            item.measurement and item.measurement.consistency_warnings
            for item in evidence
        ):
            uncertainty_parts.append(
                "The source evidence is uncertain or internally inconsistent; review the original report."
            )

        cited_results = _citation_results(evidence, retrieved)
        invalid_count = citation_issues + len(evidence) - len(cited_results)
        if invalid_count:
            uncertainty_parts.append(
                f"Citation metadata is missing or invalid for {invalid_count} selected evidence unit(s); those sources are not cited."
            )
        answer = candidate
        citations = _format_citations(
            cited_results, self.selected_document_name
        )
        if citations:
            answer = f"{answer}\n\nSources: {citations}"
        if invalid_count:
            answer = f"{answer}\n\nCitation limitation: some selected evidence lacks valid document, page, or chunk metadata."
        return GroundedAnswer(
            answer,
            cited_results,
            verification,
            " ".join(uncertainty_parts) or None,
        )

    def _no_evidence_reason(self, invalid_citations: int = 0) -> str:
        if invalid_citations:
            return (
                f"{invalid_citations} relevant source chunk(s) have missing or invalid "
                "citation metadata, so they were not used."
            )
        if self.selected_document_id is None:
            return "The source collection does not identify one selected document."
        if not self._source_evidence:
            return "No evidence belongs to the selected document."
        return "No relevant source evidence was retrieved for this question."


def _has_valid_citation(
    evidence: EvidenceUnit,
    selected_document_id: str | None,
    valid_pages: set[int] | None,
) -> bool:
    return bool(
        selected_document_id
        and evidence.document_id == selected_document_id
        and evidence.evidence_id
        and evidence.page > 0
        and (valid_pages is None or evidence.page in valid_pages)
        and (evidence.location is None or evidence.location.page == evidence.page)
        and (
            evidence.measurement is None
            or evidence.measurement.page == evidence.page
        )
    )


def _is_measurement_question(
    question: str, measurements: list[Measurement]
) -> bool:
    if (
        _asks_for_all_measurements(question)
        or _asks_for_abnormal_measurements(question)
        or _asks_within_range(question)
    ):
        return True
    return bool(
        _mentioned_measurements(question, measurements)
        or _mentions_lipid_profile(question)
        or re.search(r"\b(?:measurement|result|value|test|analyte)s?\b", question, re.I)
    )


def _asks_for_all_measurements(question: str) -> bool:
    normalized = question.casefold()
    exhaustive = re.search(
        r"\b(?:all|every|each|complete|entire)\b",
        normalized,
    )
    measurement_term = re.search(
        r"\b(?:measurements?|results?|values?|tests?|analytes?)\b",
        normalized,
    )
    return bool(
        exhaustive and measurement_term
        or re.search(
            r"\b(?:list|show|report)\b.{0,30}"
            r"\b(?:measurements?|results?|values?|tests?)\b",
            normalized,
        )
    )


def _asks_within_range(question: str) -> bool:
    normalized = question.casefold()
    return bool(
        re.search(
            r"\b(?:within|inside|in)\b.{0,60}"
            r"(?:(?:their|the|supplied|given|stated)\s+){0,3}"
            r"(?:reference\s+)?ranges?\b",
            normalized,
        )
        or re.search(
            r"\b(?:which|what)\b.{0,60}"
            r"\b(?:values|results|measurements|tests|analytes)\b"
            r".{0,60}\b(?:normal|within range|inside (?:the )?range|in range)\b",
            normalized,
        )
        or re.search(
            r"\b(?:which|what)\b.{0,50}\b(?:results|values|measurements)\b"
            r".{0,30}\bnormal\b",
            normalized,
        )
    )




def _asks_for_abnormal_measurements(question: str) -> bool:
    normalized = question.casefold()
    return bool(
        re.search(r"\b(?:abnormal|elevated|high|low)\b", normalized)
        or re.search(r"\boutside\b.{0,40}\b(?:range|ranges|reference)\b", normalized)
        or re.search(r"\bout of range\b", normalized)
    )


def _mentioned_measurements(
    question: str, measurements: list[Measurement]
) -> list[Measurement]:
    normalized_question = _normalize_name(question)
    number_tokens = set(re.findall(r"[-+]?\d+(?:\.\d+)?", question))
    selected: list[Measurement] = []
    for item in measurements:
        normalized_name = _normalize_name(item.test_name)
        aliases = {normalized_name}
        for alias in re.split(r"[,()/]", item.test_name.casefold()):
            normalized_alias = _normalize_name(alias)
            if len(normalized_alias) >= 2 and normalized_alias not in {
                "total",
                "serum",
                "plasma",
            }:
                aliases.add(normalized_alias)
        name_match = any(
            re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_question)
            for alias in aliases
            if alias
        )
        source_numbers = set(
            re.findall(r"[-+]?\d+(?:\.\d+)?", item.original_text)
        )
        if name_match or number_tokens.intersection(source_numbers):
            selected.append(item)
    return selected


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _is_lipid_measurement(name: str) -> bool:
    return any(term in name.casefold() for term in _LIPID_TERMS)


def _mentions_lipid_profile(question: str) -> bool:
    return any(term in question.casefold() for term in _LIPID_TERMS)


def _measurement_answer(
    measurements: list[Measurement], question: str
) -> str:
    if _asks_for_all_measurements(question):
        rows = "\n".join(
            f"- {_measurement_row(item)}" for item in measurements
        )
        return rows

    if _asks_within_range(question):
        flagged = [item for item in measurements if item.reported_flag]
        unflagged = [item for item in measurements if not item.reported_flag]
        parts: list[str] = []
        if unflagged:
            parts.append(
                "Within the supplied reference range: "
                + "; ".join(_measurement_fact(item) for item in unflagged)
                + "."
            )
        if flagged:
            parts.append(
                "Numerically within range with source-reported flag retained separately: "
                + "; ".join(_measurement_fact(item) for item in flagged)
                + "."
            )
        return " ".join(parts)

    abnormal_question = _asks_for_abnormal_measurements(question)
    abnormal = [
        item for item in measurements
        if classify_measurement(item).startswith(_ABNORMAL_STATUSES)
    ]
    unassessed = [
        item for item in measurements
        if classify_measurement(item) == "NOT ASSESSED"
    ]
    within = [
        item for item in measurements
        if classify_measurement(item) == "WITHIN SUPPLIED RANGE"
    ]
    if abnormal_question and abnormal:
        answer = "Reported results outside the supplied range: " + "; ".join(
            _measurement_fact(item) for item in abnormal
        ) + "."
    elif abnormal_question:
        if within:
            answer = (
                "All extracted results with an assessed range are within range: "
                + "; ".join(_measurement_fact(item) for item in within)
                + "."
            )
        else:
            answer = "Reference-range status could not be established for an extracted result."
    else:
        heading = (
            "Reported measurements"
            if _asks_for_all_measurements(question)
            else "Reported results outside the supplied range"
            if abnormal_question
            else "Relevant reported measurements"
        )
        answer = f"{heading}: " + "; ".join(
            _measurement_fact(item) for item in measurements
        ) + "."

    if unassessed and abnormal_question:
        answer += (
            " Reference-range status remains unassessed for: "
            + "; ".join(_measurement_fact(item) for item in unassessed)
            + "."
        )
    return answer


def _measurement_fact(item: Measurement) -> str:
    value = _reported_value(item)
    reference = (
        item.reference_range.raw
        if item.reference_range and item.reference_range.raw
        else "not supplied"
    )
    parts = [f"{item.test_name}: {value}", f"reference range: {reference}"]
    if item.reported_flag:
        parts.append(f"reported flag: {item.reported_flag}")
    parts.append(f"classification: {classify_measurement(item)}")
    if item.consistency_warnings:
        parts.extend(
            f"consistency warning: {warning} (derived check, distinct from source-reported values)"
            for warning in item.consistency_warnings
        )
    return "; ".join(parts)


def _measurement_row(item: Measurement) -> str:
    reference = (
        item.reference_range.raw
        if item.reference_range and item.reference_range.raw
        else "Not stated"
    )
    flag = item.reported_flag if item.reported_flag else "Not stated"
    row = (
        f"Test: {item.test_name} | "
        f"Reported value: {_reported_numeric_value(item)} | "
        f"Unit: {item.unit or 'Not stated'} | "
        f"Supplied reference range: {reference} | "
        f"Calculated classification: {classify_measurement(item)} | "
        f"Reported flag: {flag}"
    )
    if item.consistency_warnings:
        row += " | Consistency warning: " + "; ".join(
            f"{warning} (derived check, distinct from source-reported values)"
            for warning in item.consistency_warnings
        )
    return row


def _reported_numeric_value(item: Measurement) -> str:
    if item.value is None:
        return "Not stated"
    source_values = re.findall(r"[-+]?\d+(?:\.\d+)?", item.original_text)
    return next(
        (token for token in source_values if float(token) == item.value),
        f"{item.value:g}",
    )


def _reported_value(item: Measurement) -> str:
    value = _reported_numeric_value(item)
    return f"{value} {item.unit or ''}".strip()


def _extractive_answer(evidence: list[EvidenceUnit]) -> str:
    excerpts: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        excerpt = item.original_text.strip()
        if excerpt and excerpt not in seen:
            excerpts.append(excerpt)
            seen.add(excerpt)
        if len(excerpts) == 3:
            break
    return "Source excerpts: " + " ".join(excerpts)


def _has_blocking_claims(verification: list[ClaimVerification]) -> bool:
    return any(item.status in _BLOCKING_STATUSES for item in verification)


def _verification_feedback(verification: list[ClaimVerification]) -> str:
    issues = [
        warning
        for item in verification
        if item.status in _BLOCKING_STATUSES
        for warning in item.warnings
    ]
    return (
        "Revise the previous answer using only the supplied source evidence. "
        + " ".join(issues)
    )


def _citation_results(
    evidence: list[EvidenceUnit],
    retrieved: list[RetrievalResult],
) -> list[RetrievalResult]:
    by_identity = {id(item.evidence): item for item in retrieved}
    cited: list[RetrievalResult] = []
    seen: set[str] = set()
    for item in evidence:
        if not item.evidence_id or item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        result = by_identity.get(id(item))
        cited.append(
            result
            if result is not None
            else RetrievalResult(item, 0.0, ("structured-measurement",))
        )
    return cited


def _format_citations(
    results: list[RetrievalResult], document_name: str | None = None
) -> str:
    citations = []
    for item in results:
        document = (
            f"{document_name} ({item.evidence.document_id})"
            if document_name
            else item.evidence.document_id
        )
        citations.append(
            f"document {document}, p. {item.evidence.page}, "
            f"chunk {item.evidence.evidence_id}"
        )
    return "; ".join(citations)
