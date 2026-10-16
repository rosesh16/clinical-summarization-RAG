from pathlib import Path
import unittest

from chasm_rag.services.document_service import process_document
from chasm_rag.services.qa_service import answer_question
from chasm_rag.services.summary_service import create_summary


class PipelineTests(unittest.TestCase):
    def test_reference_range_question_uses_structured_thyroid_measurements(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "docs"
            / "Abnormal-Thyroid-profile-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        doc = process_document(path)

        answer = answer_question(
            doc, "Which values are outside their supplied reference ranges?"
        )

        self.assertIn("T3, TOTAL: 350.00 ng/dL", answer.answer)
        self.assertIn("reference range: 80.00 - 200.00", answer.answer)
        self.assertIn("T4, TOTAL: 28.50 mcg/dL", answer.answer)
        self.assertIn("reference range: 4.50 - 12.50", answer.answer)
        self.assertIn("TSH: 0.20 mU/L", answer.answer)
        self.assertIn("reference range: 0.40 - 4.00", answer.answer)
        self.assertIn("HIGH", answer.answer)
        self.assertIn("LOW", answer.answer)
        self.assertEqual(len(answer.evidence), 3)
        self.assertTrue(all(item.evidence.measurement for item in answer.evidence))
        self.assertFalse(answer.uncertainty)
        self.assertTrue(all(item.status.value == "SUPPORTED" for item in answer.verification))

    def test_document_to_summary_and_grounded_question(self) -> None:
        doc = process_document(Path(__file__).parents[1] / "fixtures" / "lab_report.txt")
        summary = create_summary(doc)
        answer = answer_question(doc, "What follow-up instruction is explicitly mentioned?")
        self.assertTrue(summary.output.text)
        self.assertTrue(answer.evidence)
        self.assertIn("Follow-up", answer.answer)
