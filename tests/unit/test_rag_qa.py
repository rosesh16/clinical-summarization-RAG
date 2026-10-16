from __future__ import annotations

import unittest
from uuid import uuid4

from chasm_rag.evidence.evidence_models import (
    EvidenceType,
    EvidenceUnit,
    Measurement,
    ReferenceRange,
    VerificationStatus,
)
from chasm_rag.ingestion.extraction_models import SourceLocation
from chasm_rag.qa.rag_qa import AnswerVerifier, RAGQuestionAnswerer
from chasm_rag.retrieval.embedding_service import EmbeddingService
from chasm_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from chasm_rag.verification.claim_extractor import Claim
from chasm_rag.verification.claim_verifier import ClaimVerification, ClaimVerifier


class FakeRetriever(HybridRetriever):
    def __init__(self, evidence: list[EvidenceUnit], search_ids: list[str] | None = None):
        super().__init__(evidence, embedder=EmbeddingService(enable_model=False))
        self.search_ids = search_ids
        self.search_calls: list[str] = []
        self.last_rrf_k = 60

    def search(
        self, query: str, k: int = 6, rrf_k: int = 60
    ) -> list[RetrievalResult]:
        self.search_calls.append(query)
        self.last_rrf_k = rrf_k
        selected = self.evidence
        if self.search_ids is not None:
            selected = [
                item for evidence_id in self.search_ids
                for item in self.evidence
                if item.evidence_id == evidence_id
            ]
        return [RetrievalResult(item, 1.0, ("fake",)) for item in selected[:k]]


class RecordingVerifier:
    def __init__(self, retriever: FakeRetriever):
        self.delegate = ClaimVerifier(retriever)
        self.calls: list[tuple[str, list[EvidenceUnit]]] = []

    def verify(
        self, text: str, candidate_evidence: list[EvidenceUnit] | None = None
    ) -> list[ClaimVerification]:
        evidence = candidate_evidence if candidate_evidence is not None else []
        self.calls.append((text, evidence))
        return self.delegate.verify(text, candidate_evidence)


class MockAnswerGenerator:
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.calls: list[tuple[str, list[EvidenceUnit], str | None]] = []

    def generate(
        self,
        question: str,
        evidence: list[EvidenceUnit],
        feedback: str | None = None,
    ) -> str:
        self.calls.append((question, evidence, feedback))
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


class AlwaysContradictVerifier:
    def __init__(self):
        self.calls: list[tuple[str, list[EvidenceUnit]]] = []

    def verify(
        self, text: str, candidate_evidence: list[EvidenceUnit] | None = None
    ) -> list[ClaimVerification]:
        evidence = candidate_evidence or []
        self.calls.append((text, evidence))
        return [
            ClaimVerification(
                Claim("claim-1", text),
                VerificationStatus.CONTRADICTED,
                [item.evidence_id for item in evidence],
                [item.original_text for item in evidence],
                "test verifier",
                ["Conflicting source claims."],
            )
        ]


def make_evidence(
    text: str,
    *,
    document_id: str = "doc-a",
    page: int = 1,
    evidence_id: str | None = None,
    location_page: int | None = None,
    measurement: Measurement | None = None,
) -> EvidenceUnit:
    unit = EvidenceUnit(
        original_text=text,
        normalized_text=text.casefold(),
        document_id=document_id,
        page=page,
        evidence_type=EvidenceType.MEASUREMENT if measurement else EvidenceType.NARRATIVE,
        location=SourceLocation(location_page if location_page is not None else page),
        extraction_confidence=0.95,
        measurement=measurement,
        evidence_id=evidence_id or str(uuid4()),
    )
    if measurement:
        measurement.evidence_id = unit.evidence_id
    return unit


def make_measurement(
    name: str,
    value: float,
    unit: str,
    lower: float,
    upper: float,
    *,
    flag: str | None = None,
    source_text: str | None = None,
    page: int = 1,
    warnings: list[str] | None = None,
) -> EvidenceUnit:
    raw = f"{lower:g}-{upper:g}"
    text = source_text or (
        f"{name} | {value:g} | {flag or ''} | {raw} | {unit}"
    )
    measurement = Measurement(
        test_name=name,
        value=value,
        unit=unit,
        reference_range=ReferenceRange(lower=lower, upper=upper, raw=raw),
        reported_flag=flag,
        page=page,
        original_text=text,
        consistency_warnings=list(warnings or []),
    )
    return make_evidence(text, page=page, measurement=measurement)


def make_answerer(
    evidence: list[EvidenceUnit],
    *,
    search_ids: list[str] | None = None,
    document_id: str = "doc-a",
    generator: MockAnswerGenerator | None = None,
    verifier: AnswerVerifier | None = None,
) -> tuple[RAGQuestionAnswerer, FakeRetriever, AnswerVerifier]:
    retriever = FakeRetriever(evidence, search_ids)
    actual_verifier = verifier or RecordingVerifier(retriever)
    answerer = RAGQuestionAnswerer(
        retriever,
        actual_verifier,
        answer_generator=generator,
        selected_document_id=document_id,
        valid_pages={1, 2},
    )
    return answerer, retriever, actual_verifier


class RAGQuestionAnswererTests(unittest.TestCase):
    def test_lipid_question_returns_complete_structured_panel(self) -> None:
        panel = [
            make_measurement("Total Cholesterol", 250, "mg/dL", 0, 200, flag="HIGH"),
            make_measurement("HDL Cholesterol", 55, "mg/dL", 40, 60),
            make_measurement("LDL Cholesterol", 165, "mg/dL", 0, 130, flag="HIGH"),
            make_measurement("Triglycerides", 120, "mg/dL", 0, 150),
        ]
        answerer, _, _ = make_answerer(panel)

        answer = answerer.answer("What are all lipid measurements?")

        for test in ("Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides"):
            self.assertIn(test, answer.answer)
        self.assertIn("Reported flag: HIGH", answer.answer)
        self.assertEqual(len(answer.evidence), 4)
        self.assertIsNone(answer.uncertainty)

    def test_all_measurements_are_formatted_as_rows_with_missing_flags(self) -> None:
        sodium = make_measurement("Sodium", 140, "mmol/L", 135, 145)
        potassium = make_measurement(
            "Potassium", 4.2, "mmol/L", 3.5, 5.1, flag="HIGH"
        )
        answerer, _, _ = make_answerer([sodium, potassium])

        answer = answerer.answer(
            "List all reported measurements with their values, units, "
            "reference ranges, and reported flags."
        )

        rows = answer.answer.split("\n\nSources:")[0].splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn(
            "- Test: Sodium | Reported value: 140 | Unit: mmol/L | "
            "Supplied reference range: 135-145 | "
            "Calculated classification: WITHIN SUPPLIED RANGE | "
            "Reported flag: Not stated",
            rows[0],
        )
        self.assertIn("Test: Potassium", rows[1])
        self.assertIn("Reported flag: HIGH", rows[1])

    def test_all_measurements_preserve_borderline_flag_separately(self) -> None:
        platelet = make_measurement(
            "Platelet Count",
            150000,
            "/uL",
            150000,
            410000,
            flag="Borderline",
        )
        answerer, _, _ = make_answerer([platelet])

        answer = answerer.answer(
            "List all reported measurements with their values, units, "
            "reference ranges, and reported flags."
        )

        rows = answer.answer.split("\n\nSources:")[0].splitlines()
        self.assertEqual(len(rows), 1)
        self.assertIn("Test: Platelet Count", rows[0])
        self.assertIn("Reported value: 150000", rows[0])
        self.assertIn("Unit: /uL", rows[0])
        self.assertIn("Supplied reference range: 150000-410000", rows[0])
        self.assertIn(
            "Calculated classification: WITHIN SUPPLIED RANGE",
            rows[0],
        )
        self.assertIn("Reported flag: Borderline", rows[0])
        self.assertNotIn("Reported flag: NORMAL", rows[0])

    def test_non_lipid_abnormal_question_uses_all_thyroid_measurements(self) -> None:
        evidence = [
            make_measurement("T3, TOTAL", 350, "ng/dL", 80, 200, flag="HIGH"),
            make_measurement("T4, TOTAL", 28.5, "mcg/dL", 4.5, 12.5, flag="HIGH"),
            make_measurement("TSH", 0.2, "mU/L", 0.4, 4.0, flag="LOW"),
        ]
        answerer, _, _ = make_answerer(evidence, search_ids=[evidence[0].evidence_id])

        answer = answerer.answer("Which values are outside their supplied ranges?")

        self.assertIn("T3, TOTAL: 350 ng/dL", answer.answer)
        self.assertIn("T4, TOTAL: 28.5 mcg/dL", answer.answer)
        self.assertIn("TSH: 0.2 mU/L", answer.answer)
        self.assertEqual(len(answer.evidence), 3)
        self.assertTrue(all(item.evidence.measurement for item in answer.evidence))

    def test_general_question_uses_injected_question_evidence_generator(self) -> None:
        source = make_evidence("Follow-up with the clinic in two weeks.")
        generator = MockAnswerGenerator(["Follow-up is with the clinic in two weeks."])
        answerer, _, verifier = make_answerer([source], generator=generator)

        answer = answerer.answer("What follow-up is mentioned?")

        self.assertIn("Follow-up is with the clinic", answer.answer)
        self.assertEqual(generator.calls[0][0], "What follow-up is mentioned?")
        self.assertEqual(generator.calls[0][1], [source])
        self.assertIn("document doc-a, p. 1", answer.answer)
        self.assertIn(f"chunk {source.evidence_id}", answer.answer)
        self.assertIsInstance(verifier, RecordingVerifier)
        if not isinstance(verifier, RecordingVerifier):
            self.fail("Expected recording verifier")
        self.assertEqual(verifier.calls[0][1], [source])

    def test_exact_numerical_lookup_preserves_reported_measurement(self) -> None:
        potassium = make_measurement(
            "Potassium",
            6.8,
            "mmol/L",
            3.5,
            5.1,
            flag="HIGH",
            source_text="Potassium: 6.8 mmol/L Reference Range: 3.5-5.1 HIGH",
        )
        answerer, _, _ = make_answerer([potassium])

        answer = answerer.answer("What is the potassium result?")

        self.assertIn("Potassium: 6.8 mmol/L", answer.answer)
        self.assertIn("reference range: 3.5-5.1", answer.answer)
        self.assertIn("reported flag: HIGH", answer.answer)

    def test_all_abnormal_measurements_uses_complete_store_not_top_k(self) -> None:
        evidence = [
            make_measurement("Sodium", 140, "mmol/L", 135, 145),
            make_measurement("Potassium", 6.8, "mmol/L", 3.5, 5.1, flag="HIGH"),
            make_measurement("Glucose", 45, "mg/dL", 70, 110, flag="LOW"),
        ]
        answerer, _, _ = make_answerer(evidence, search_ids=[evidence[0].evidence_id])

        answer = answerer.answer("List all abnormal measurements.")

        self.assertNotIn("Sodium", answer.answer)
        self.assertIn("Test: Potassium", answer.answer)
        self.assertIn("Reported value: 6.8", answer.answer)
        self.assertIn("Test: Glucose", answer.answer)
        self.assertIn("Reported value: 45", answer.answer)
        self.assertEqual(
            {item.evidence.evidence_id for item in answer.evidence},
            {evidence[1].evidence_id, evidence[2].evidence_id},
        )

    def test_abnormal_question_reports_when_all_measurements_are_within_range(self) -> None:
        evidence = [
            make_measurement("Sodium", 140, "mmol/L", 135, 145),
            make_measurement("Potassium", 4.2, "mmol/L", 3.5, 5.1),
        ]
        answerer, _, _ = make_answerer(evidence, search_ids=[evidence[0].evidence_id])

        answer = answerer.answer("Which results are abnormal?")

        self.assertIn("All extracted results with an assessed range are within range", answer.answer)
        self.assertIn("Sodium: 140 mmol/L", answer.answer)
        self.assertIn("Potassium: 4.2 mmol/L", answer.answer)
        self.assertEqual(len(answer.evidence), 2)

    def test_within_range_question_uses_structured_measurements(self) -> None:
        within = make_measurement("Sodium", 140, "mmol/L", 135, 145)
        abnormal = make_measurement(
            "Potassium", 6.8, "mmol/L", 3.5, 5.1, flag="HIGH"
        )
        header = make_evidence("Investigation Result Reference Value Unit")
        answerer, _, _ = make_answerer(
            [within, abnormal, header], search_ids=[header.evidence_id]
        )

        for question in (
            "Which values are within their supplied reference ranges?",
            "Which results are normal?",
            "Which measurements are inside the reference range?",
        ):
            with self.subTest(question=question):
                answer = answerer.answer(question)
                self.assertIn("Sodium: 140 mmol/L", answer.answer)
                self.assertNotIn("Potassium", answer.answer)
                self.assertTrue(all(item.evidence.measurement for item in answer.evidence))
                self.assertEqual(
                    {item.evidence.evidence_id for item in answer.evidence},
                    {within.evidence_id},
                )

    def test_normal_results_are_selected_by_numeric_reference_range(self) -> None:
        within = make_measurement("Hemoglobin", 14, "g/dL", 12, 16)
        low = make_measurement("PCV", 30, "%", 36, 46, flag="LOW")
        answerer, _, _ = make_answerer(
            [within, low], search_ids=[low.evidence_id]
        )

        answer = answerer.answer("Which results are normal?")

        self.assertIn("Hemoglobin: 14 g/dL", answer.answer)
        self.assertNotIn("PCV", answer.answer)
        self.assertIn("classification: WITHIN SUPPLIED RANGE", answer.answer)
        self.assertEqual(
            [item.evidence.evidence_id for item in answer.evidence],
            [within.evidence_id],
        )

    def test_borderline_source_flag_is_separate_at_inclusive_range_boundary(self) -> None:
        platelet = make_measurement(
            "Platelet Count",
            150000,
            "/uL",
            150000,
            410000,
            flag="Borderline",
        )
        answerer, _, _ = make_answerer([platelet])

        answer = answerer.answer(
            "Which values are within their supplied reference ranges?"
        )

        self.assertIn(
            "Numerically within range with source-reported flag retained separately",
            answer.answer,
        )
        self.assertIn("Platelet Count: 150000 /uL", answer.answer)
        self.assertIn("reference range: 150000-410000", answer.answer)
        self.assertIn("reported flag: Borderline", answer.answer)
        self.assertIn("classification: WITHIN SUPPLIED RANGE", answer.answer)
        self.assertNotIn("classification: NORMAL", answer.answer)

    def test_within_range_question_without_structured_measurements_is_explicit(self) -> None:
        header = make_evidence("Investigation Result Reference Value Unit End of Report")
        generator = MockAnswerGenerator(["Irrelevant retrieved report header."])
        answerer, _, _ = make_answerer([header], generator=generator)

        answer = answerer.answer(
            "Which values are within their supplied reference ranges?"
        )

        self.assertIn("contains no structured measurements", answer.answer)
        self.assertTrue(answer.uncertainty)
        self.assertEqual(generator.calls, [])
        self.assertEqual(answer.evidence, [])

    def test_no_supporting_evidence_returns_clear_uncertainty(self) -> None:
        evidence = [make_evidence("The report lists a sample collection date.")]
        answerer, retriever, _ = make_answerer(evidence, search_ids=[])

        answer = answerer.answer("What medication was prescribed?")

        self.assertIn("does not contain usable evidence", answer.answer)
        self.assertTrue(answer.uncertainty)
        self.assertEqual(retriever.search_calls, ["What medication was prescribed?"])

    def test_fabricated_number_retries_then_uses_verified_extract(self) -> None:
        source = make_evidence("The clinic follow-up is scheduled in 2 weeks.")
        generator = MockAnswerGenerator(
            [
                "The follow-up is scheduled in 999 weeks.",
                "The follow-up is scheduled in 2 weeks.",
            ]
        )
        answerer, _, _ = make_answerer([source], generator=generator)

        answer = answerer.answer("When is the follow-up?")

        self.assertEqual(len(generator.calls), 2)
        self.assertIsNotNone(generator.calls[1][2])
        self.assertIn("2 weeks", answer.answer)
        self.assertNotIn("999", answer.answer)
        self.assertIsNone(answer.uncertainty)

    def test_contradicted_answer_is_not_returned_as_verified(self) -> None:
        source = make_evidence("Potassium was reported as 4.2 mmol/L.")
        generator = MockAnswerGenerator(["Potassium was reported as 999 mmol/L."])
        verifier = AlwaysContradictVerifier()
        answerer, _, _ = make_answerer(
            [source], generator=generator, verifier=verifier
        )

        answer = answerer.answer("Summarize this note.")

        self.assertIsInstance(verifier, AlwaysContradictVerifier)
        if not isinstance(verifier, AlwaysContradictVerifier):
            self.fail("Expected contradictory verifier")
        self.assertNotIn("999 mmol/L", answer.answer)
        self.assertIn("could not be verified", answer.answer)
        self.assertTrue(answer.uncertainty)
        self.assertEqual(len(verifier.calls), 3)
        self.assertTrue(all(call[1] == [source] for call in verifier.calls))

    def test_invalid_citation_metadata_is_not_fabricated(self) -> None:
        invalid_sources = [
            make_evidence("The result is 6.8 mmol/L.", page=0, location_page=0),
            make_evidence("The result is 6.8 mmol/L.", page=1, location_page=2),
        ]
        for invalid in invalid_sources:
            with self.subTest(page=invalid.page, location=invalid.location):
                answerer, _, _ = make_answerer([invalid])

                answer = answerer.answer("What was the result?")

                self.assertFalse(answer.evidence)
                self.assertTrue(answer.uncertainty)
                self.assertIn("citation metadata", answer.uncertainty or "")
                self.assertNotIn("Sources:", answer.answer)

    def test_cross_document_retrieval_is_filtered_to_selected_document(self) -> None:
        selected = make_evidence(
            "Selected document: follow-up in 2 weeks.", document_id="doc-a"
        )
        other = make_evidence(
            "Other document: follow-up in 99 weeks.", document_id="doc-b"
        )
        generator = MockAnswerGenerator(["Follow-up is in 2 weeks."])
        answerer, _, verifier = make_answerer(
            [other, selected],
            search_ids=[other.evidence_id, selected.evidence_id],
            generator=generator,
        )

        answer = answerer.answer("When is the follow-up?")

        self.assertNotIn("99 weeks", answer.answer)
        self.assertTrue(all(item.evidence.document_id == "doc-a" for item in answer.evidence))
        self.assertEqual(generator.calls[0][1], [selected])
        self.assertIsInstance(verifier, RecordingVerifier)
        if not isinstance(verifier, RecordingVerifier):
            self.fail("Expected recording verifier")
        self.assertEqual(verifier.calls[0][1], [selected])

    def test_inconsistent_calculation_preserves_reported_values_and_warning(self) -> None:
        warning = (
            "Derived check: total cholesterol (250) - HDL cholesterol (55) = "
            "195 mg/dL, which differs from the reported non-HDL value (180 mg/dL)."
        )
        evidence = [
            make_measurement("Total Cholesterol", 250, "mg/dL", 0, 200, flag="HIGH"),
            make_measurement("HDL Cholesterol", 55, "mg/dL", 40, 60),
            make_measurement(
                "Non-HDL Cholesterol",
                180,
                "mg/dL",
                0,
                160,
                warnings=[warning],
            ),
        ]
        answerer, _, _ = make_answerer(evidence)

        answer = answerer.answer("What are all lipid measurements?")

        self.assertIn("Test: Non-HDL Cholesterol", answer.answer)
        self.assertIn("Reported value: 180", answer.answer)
        self.assertIn("= 195 mg/dL", answer.answer)
        self.assertIn("derived check, distinct from source-reported values", answer.answer)
        self.assertTrue(answer.uncertainty)
        self.assertIsNotNone(answer.uncertainty)
        self.assertIn("internally inconsistent", answer.uncertainty or "")

    def test_contradictory_duplicate_values_are_both_retained(self) -> None:
        warning = "Repeated measurement has conflicting reported values; preserve both source values."
        evidence = [
            make_measurement(
                "Potassium",
                4.2,
                "mmol/L",
                3.5,
                5.1,
                source_text="Potassium: 4.2 mmol/L Reference Range: 3.5-5.1",
                warnings=[warning],
            ),
            make_measurement(
                "Potassium",
                6.8,
                "mmol/L",
                3.5,
                5.1,
                flag="HIGH",
                source_text="Potassium: 6.8 mmol/L Reference Range: 3.5-5.1 HIGH",
                page=2,
                warnings=[warning],
            ),
        ]
        answerer, _, _ = make_answerer(evidence)

        answer = answerer.answer("List all measurements.")

        self.assertEqual(answer.answer.count("Test: Potassium"), 2)
        self.assertIn("Reported value: 4.2", answer.answer)
        self.assertIn("Reported value: 6.8", answer.answer)
        self.assertIn("conflicting reported values", answer.answer)
        self.assertTrue(answer.uncertainty)


if __name__ == "__main__":
    unittest.main()
