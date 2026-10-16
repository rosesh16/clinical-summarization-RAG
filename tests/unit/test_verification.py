from pathlib import Path
import unittest

from chasm_rag.services.document_service import process_document
from chasm_rag.summarization.generator import StaticGenerator
from chasm_rag.summarization.chasm_pipeline import ChasmPipeline
from chasm_rag.services.summary_service import create_summary


class VerificationTests(unittest.TestCase):
    def test_thyroid_fallback_summarizes_pattern_and_report_caveats(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "docs"
            / "Abnormal-Thyroid-profile-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        doc = process_document(path)
        result = create_summary(
            doc,
            ChasmPipeline(StaticGenerator("An unsupported claim about the patient.")),
        )

        self.assertTrue(result.output.used_fallback)
        self.assertIn("total T3 at 350 ng/dL", result.output.text)
        self.assertIn("above the supplied reference range", result.output.text)
        self.assertIn("total T4 at 28.5 mcg/dL", result.output.text)
        self.assertIn("TSH at 0.2 mU/L", result.output.text)
        self.assertIn("biochemical pattern commonly associated with hyperthyroidism", result.output.text)
        self.assertIn("not by itself a confirmed clinical diagnosis", result.output.text)
        self.assertIn("TSH levels vary over the course of the day", result.output.text)
        self.assertIn("pregnancy or steroid therapy", result.output.text)
        self.assertIn("free T3 and free T4 may correlate more closely", result.output.text)
        self.assertNotIn("T3, TOTAL:", result.output.text)

    def test_unsupported_injected_claim_triggers_fallback(self) -> None:
        doc = process_document(Path(__file__).parents[1] / "fixtures" / "lab_report.txt")
        result = create_summary(doc, ChasmPipeline(StaticGenerator("LDL Cholesterol is 999 mg/dL.")))
        self.assertTrue(result.output.used_fallback)
        self.assertIn("The lipid profile reports", result.output.text)

    def test_lipid_fallback_lists_every_result_and_keeps_consistency_check_distinct(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "PDFs"
            / "LIPID-PROFILE-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        doc = process_document(path)
        result = create_summary(
            doc,
            ChasmPipeline(StaticGenerator("Unsupported result: LDL is 999 mg/dL.")),
        )

        self.assertTrue(result.output.used_fallback)
        for name in (
            "Cholesterol Total",
            "Triglycerides",
            "HDL Cholesterol",
            "LDL Cholesterol",
            "VLDL Cholesterol",
            "Non-HDL Cholesterol",
        ):
            self.assertIn(name, result.output.text)
        self.assertIn("calculated classification: HIGH (above supplied threshold)", result.output.text)
        self.assertIn("reported flag: High", result.output.text)
        self.assertIn("reported value 250.00 mg/dL", result.output.text)
        self.assertIn("[p. 1]", result.output.text)
        self.assertIn("Derived check from report values", result.output.text)
        self.assertIn("reported non-HDL value", result.output.text)

    def test_normal_source_flags_do_not_force_a_fallback_when_draft_is_complete(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "PDFs"
            / "Normal-CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        doc = process_document(path)
        result = create_summary(
            doc,
            ChasmPipeline(
                StaticGenerator(
                    "The report lists Hemoglobin, Total RBC count, Packed Cell Volume, "
                    "Mean Corpuscular Volume, MCH, MCHC, RDW, Total WBC count, "
                    "Neutrophils, Lymphocytes, Eosinophils, Monocytes, Basophils, "
                    "and Platelet Count at 20000 cumm."
                )
            ),
        )

        self.assertTrue(result.output.used_fallback)
        self.assertIn("Source normal flag conflicts", result.output.text)
        self.assertIn("Platelet Count: reported value 20000 cumm", result.output.text)
        self.assertIn("calculated classification: LOW (below supplied range)", result.output.text)
        self.assertIn("reported flag: Normal", result.output.text)
