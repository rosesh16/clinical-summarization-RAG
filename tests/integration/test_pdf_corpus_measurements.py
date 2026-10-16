from pathlib import Path
import unittest

from chasm_rag.services.document_service import process_document


PDF_DIR = Path(__file__).parents[2] / "PDFs"
EXPECTED_COUNTS = {
    "Abnormal-Thyroid-profile-test-report-format-example-sample-template-Drlogy-lab-report.pdf": 3,
    "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf": 14,
    "kft-test-normal-report-format-example-sample-template-drlogy-lab-report.pdf": 11,
    "LIPID-PROFILE-test-report-format-example-sample-template-Drlogy-lab-report.pdf": 6,
    "Normal-CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf": 14,
    "Normal-Lipid-profile-test-report-format-example-sample-template-Drlogy-lab-report.pdf": 6,
}


class PdfCorpusMeasurementTests(unittest.TestCase):
    def test_all_repository_lab_pdfs_extract_expected_measurement_counts(self) -> None:
        for filename, expected_count in EXPECTED_COUNTS.items():
            with self.subTest(filename=filename):
                document = process_document(PDF_DIR / filename)
                self.assertEqual(len(document.store.measurements), expected_count)
                self.assertNotIn(
                    "EIA",
                    {measurement.test_name for measurement in document.store.measurements},
                )

    def test_corpus_preserves_reported_out_of_range_values_and_source_flags(self) -> None:
        normal_cbc = process_document(
            PDF_DIR
            / "Normal-CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        platelet = next(
            item
            for item in normal_cbc.store.measurements
            if item.test_name == "Platelet Count"
        )
        self.assertEqual((platelet.value, platelet.unit, platelet.reported_flag), (20000, "cumm", "Normal"))
        self.assertIn(
            "Source normal flag conflicts",
            " ".join(platelet.consistency_warnings),
        )

        kft = process_document(
            PDF_DIR / "kft-test-normal-report-format-example-sample-template-drlogy-lab-report.pdf"
        )
        sodium = next(item for item in kft.store.measurements if item.test_name == "Sodium")
        self.assertEqual((sodium.value, sodium.unit, sodium.reported_flag), (100, "mEq/L", "Normal"))
        self.assertIn(
            "Source normal flag conflicts",
            " ".join(sodium.consistency_warnings),
        )

    def test_lipid_recommendation_chart_is_not_parsed_as_patient_measurements(self) -> None:
        document = process_document(
            PDF_DIR
            / "LIPID-PROFILE-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        )
        names = {item.test_name for item in document.store.measurements}
        self.assertEqual(
            names,
            {
                "Cholesterol Total",
                "Triglycerides",
                "HDL Cholesterol",
                "LDL Cholesterol",
                "VLDL Cholesterol",
                "Non-HDL Cholesterol",
            },
        )
