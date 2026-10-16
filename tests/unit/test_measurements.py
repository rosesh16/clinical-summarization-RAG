from pathlib import Path
import unittest

from chasm_rag.evidence.evidence_store import build_evidence_store
from chasm_rag.evidence.measurement_parser import parse_measurement_line
from chasm_rag.evidence.measurement_validator import (
    classify_measurement,
    validate_measurements,
)
from chasm_rag.ingestion.document_loader import extract_document
from chasm_rag.ingestion.extraction_models import (
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
)


class MeasurementTests(unittest.TestCase):
    def test_preserves_value_unit_range_and_flag(self) -> None:
        path = Path(__file__).parents[1] / "fixtures" / "lab_report.txt"
        store = build_evidence_store(extract_document(path))
        cholesterol = next(item for item in store.measurements if item.test_name == "Total Cholesterol")
        self.assertEqual(cholesterol.value, 250)
        self.assertEqual(cholesterol.unit, "mg/dL")
        self.assertEqual(cholesterol.reference_range.upper, 200)
        self.assertEqual(cholesterol.reported_flag, "HIGH")

    def test_flags_conflicting_repeated_measurements(self) -> None:
        path = Path(__file__).parents[1] / "fixtures" / "conflicting_report.txt"
        store = build_evidence_store(extract_document(path))
        self.assertTrue(all(item.consistency_warnings for item in store.measurements))

    def test_parses_flag_before_reference_and_unit_after_reference(self) -> None:
        measurement = parse_measurement_line(
            "T3, TOTAL   350.00   High   80.00 - 200.00   ng/dL",
            1,
        )
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.test_name, "T3, TOTAL")
        self.assertEqual(measurement.value, 350)
        self.assertEqual(measurement.unit, "ng/dL")
        self.assertEqual(measurement.reference_range.raw, "80.00 - 200.00")
        self.assertEqual(measurement.reported_flag, "High")

    def test_parses_ocr_table_row_with_single_space_separators(self) -> None:
        measurement = parse_measurement_line(
            "T3, TOTAL 350.00 High 80.00 - 200.00 ng/dL",
            1,
        )
        self.assertIsNotNone(measurement)
        self.assertEqual(
            (measurement.test_name, measurement.value, measurement.unit, measurement.reported_flag),
            ("T3, TOTAL", 350, "ng/dL", "High"),
        )
        self.assertIsNone(
            parse_measurement_line("Sample Type Serum (2 ml) TAT: 1 day (Normal: 1 - 3 days)", 1)
        )
        self.assertIsNone(
            parse_measurement_line("To Check Report Authenticity by Scanning QR Code on Top e Generated on: 02 Dec", 1)
        )

    def test_parses_qualitative_flags_without_using_them_as_units(self) -> None:
        cases = (
            ("HIGH", "HIGH"),
            ("LOW", "LOW"),
            ("ABNORMAL", "ABNORMAL"),
            ("High", "High"),
            ("h", "h"),
            ("H", "H"),
            ("L", "L"),
            ("Borderline", "Borderline"),
        )
        for source_flag, expected_flag in cases:
            with self.subTest(source_flag=source_flag):
                measurement = parse_measurement_line(
                    f"Platelet Count 150000 {source_flag} "
                    "150000 - 410000 cumm",
                    1,
                )
                self.assertIsNotNone(measurement)
                self.assertEqual(measurement.value, 150000)
                self.assertEqual(measurement.unit, "cumm")
                self.assertEqual(measurement.reference_range.raw, "150000 - 410000")
                self.assertEqual(measurement.reported_flag, expected_flag)

    def test_parses_measurement_without_a_reported_flag(self) -> None:
        measurement = parse_measurement_line(
            "Platelet Count 150000 cumm 150000 - 410000",
            1,
        )
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.unit, "cumm")
        self.assertIsNone(measurement.reported_flag)

    def test_parses_reference_backed_result_without_unit_or_flag(self) -> None:
        measurement = parse_measurement_line(
            "Sodium: 140 Reference Range: 135 - 145",
            1,
        )
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.value, 140)
        self.assertIsNone(measurement.unit)
        self.assertIsNone(measurement.reported_flag)

    def test_cbc_table_layout_keeps_borderline_flag_separate_from_unit(self) -> None:
        raw_row = [
            "Platelet Count",
            "150000",
            "Borderline",
            "150000 - 410000",
            "cumm",
        ]
        document = ExtractedDocument(
            source_path=Path("cbc-report.pdf"),
            source_filename="cbc-report.pdf",
            pages=[
                ExtractedPage(
                    page=1,
                    text="",
                    method="text",
                    tables=[
                        ExtractedTable(
                            page=1,
                            rows=[
                                ["Investigation", "Result", "Unit", "Reference Range"],
                                raw_row,
                            ],
                            name="CBC",
                        )
                    ],
                )
            ],
        )

        store = build_evidence_store(document)

        platelet = next(
            item for item in store.measurements
            if item.test_name == "Platelet Count"
        )
        self.assertEqual(platelet.value, 150000)
        self.assertEqual(platelet.unit, "cumm")
        self.assertEqual(platelet.reference_range.raw, "150000 - 410000")
        self.assertEqual(platelet.reported_flag, "Borderline")
        self.assertEqual(classify_measurement(platelet), "WITHIN SUPPLIED RANGE")
        self.assertEqual(platelet.original_text, " | ".join(raw_row))
        self.assertEqual(
            store.by_id(platelet.evidence_id).original_text,
            " | ".join(raw_row),
        )

    def test_fragmented_column_table_reconstructs_measurements(self) -> None:
        document = ExtractedDocument(
            source_path=Path("cbc-columns.pdf"),
            source_filename="cbc-columns.pdf",
            pages=[
                ExtractedPage(
                    page=1,
                    text="",
                    method="text",
                    tables=[
                        ExtractedTable(
                            page=1,
                            rows=[
                                ["Investigation", "Platelet Count", "WBC"],
                                ["Result", "150000", "7.2"],
                                ["Unit", "cumm", "thou/uL"],
                                ["Reference Range", "150000 - 410000", "4.0 - 11.0"],
                                ["Flag", "Borderline", ""],
                            ],
                            name="CBC",
                        )
                    ],
                )
            ],
        )

        measurements = {
            item.test_name: item for item in build_evidence_store(document).measurements
        }

        self.assertEqual(measurements["Platelet Count"].unit, "cumm")
        self.assertEqual(measurements["Platelet Count"].reported_flag, "Borderline")
        self.assertEqual(measurements["WBC"].unit, "thou/uL")
        self.assertIsNone(measurements["WBC"].reported_flag)
        self.assertIn(
            "Investigation: Platelet Count",
            measurements["Platelet Count"].original_text,
        )

    def test_table_rows_preserve_high_low_and_absent_flags(self) -> None:
        rows = [
            ["Hemoglobin", "17.2", "g/dL", "12.0 - 16.0", "HIGH"],
            ["WBC", "3.0", "thou/uL", "4.0 - 11.0", "LOW"],
            ["RBC", "4.5", "mil/uL", "4.0 - 5.5", ""],
        ]
        document = ExtractedDocument(
            source_path=Path("cbc-report.pdf"),
            source_filename="cbc-report.pdf",
            pages=[
                ExtractedPage(
                    page=1,
                    text="",
                    method="text",
                    tables=[
                        ExtractedTable(
                            page=1,
                            rows=[
                                [
                                    "Investigation", "Result", "Unit",
                                    "Reference Range", "Flag",
                                ],
                                *rows,
                            ],
                        )
                    ],
                )
            ],
        )

        measurements = {
            item.test_name: item for item in build_evidence_store(document).measurements
        }

        self.assertEqual(measurements["Hemoglobin"].reported_flag, "HIGH")
        self.assertEqual(measurements["WBC"].reported_flag, "LOW")
        self.assertIsNone(measurements["RBC"].reported_flag)
        self.assertEqual(measurements["RBC"].unit, "mil/uL")

    def test_lipid_pdf_cells_separate_normal_flags_and_ignore_nla_ranges(self) -> None:
        lipid_rows = [
            ["Cholesterol Total", "150.00\nNormal", "< 200.00", "mg/dL"],
            ["Triglycerides", "100.00\nNormal", "< 150.00", "mg/dL"],
            ["HDL Cholesterol", "50.00\nNormal", "> 40.00", "mg/dL"],
            ["LDL Cholesterol", "80.00\nNormal", "< 100.00", "mg/dL"],
            ["VLDL Cholesterol", "20.00\nNormal", "< 30.00", "mg/dL"],
            ["Non-HDL Cholesterol", "100.00\nNormal", "< 130.00", "mg/dL"],
        ]
        document = ExtractedDocument(
            source_path=Path("lipid-report.pdf"),
            source_filename="lipid-report.pdf",
            pages=[
                ExtractedPage(
                    page=1,
                    text=(
                        "NLA - 2014 RECOMMENDATIONS\n"
                        "Borderline High | 200 - 239 | | 130 - 159 | 150 - 199\n"
                        "NLA - 2014"
                    ),
                    method="text",
                    tables=[
                        ExtractedTable(
                            page=1,
                            rows=[
                                ["Investigation", "Result", "Reference Value", "Unit"],
                                *lipid_rows,
                            ],
                            name="Lipid profile",
                        ),
                        ExtractedTable(
                            page=1,
                            rows=[
                                ["NLA - 2014 RECOMMENDATIONS", "Total Cholesterol"],
                                ["Borderline High", "200 - 239"],
                            ],
                            name="Recommendations",
                        ),
                    ],
                )
            ],
        )

        measurements = build_evidence_store(document).measurements
        lipid_measurements = [
            item for item in measurements
            if item.test_name in {
                "Cholesterol Total", "Triglycerides", "HDL Cholesterol",
                "LDL Cholesterol", "VLDL Cholesterol", "Non-HDL Cholesterol",
            }
        ]

        self.assertEqual(len(lipid_measurements), 6)
        for item in lipid_measurements:
            with self.subTest(test=item.test_name):
                self.assertEqual(item.unit, "mg/dL")
                self.assertEqual(item.reported_flag, "Normal")
                self.assertIsNotNone(item.reference_range)
                self.assertEqual(
                    classify_measurement(item),
                    "WITHIN SUPPLIED RANGE",
                )
        self.assertFalse(
            any(
                item.test_name in {"Borderline High", "NLA -"}
                for item in measurements
            )
        )

    def test_flat_lipid_result_line_retains_normal_flag_and_reported_unit(self) -> None:
        measurement = parse_measurement_line(
            "Cholesterol Total | 150.00 | Normal | < 200.00 | mg/dL",
            1,
        )

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.value, 150)
        self.assertEqual(measurement.unit, "mg/dL")
        self.assertEqual(measurement.reference_range.raw, "< 200.00")
        self.assertEqual(measurement.reported_flag, "Normal")

    def test_threshold_flags_are_checked_against_strict_and_inclusive_limits(self) -> None:
        cholesterol = parse_measurement_line(
            "Cholesterol Total 250 HIGH < 200 mg/dL",
            1,
        )
        at_threshold = parse_measurement_line(
            "Cholesterol Total 200 <= 200 mg/dL",
            1,
        )

        self.assertIsNotNone(cholesterol)
        self.assertIsNotNone(at_threshold)
        validate_measurements([cholesterol])
        validate_measurements([at_threshold])
        self.assertEqual(cholesterol.consistency_warnings, [])
        self.assertEqual(at_threshold.consistency_warnings, [])
        self.assertEqual(classify_measurement(cholesterol), "HIGH (above supplied threshold)")
        self.assertEqual(classify_measurement(at_threshold), "WITHIN SUPPLIED RANGE")

    def test_normal_source_flag_is_preserved_when_numeric_value_is_outside_range(self) -> None:
        sodium = parse_measurement_line(
            "Sodium 100 Normal 135 - 145 mEq/L",
            1,
        )

        self.assertIsNotNone(sodium)
        validate_measurements([sodium])
        self.assertEqual(sodium.reported_flag, "Normal")
        self.assertEqual(classify_measurement(sodium), "LOW (below supplied range)")
        self.assertTrue(
            any("Source normal flag conflicts" in warning for warning in sodium.consistency_warnings)
        )

    def test_extracts_measurements_from_thyroid_report_pdf(self) -> None:
        path = Path(__file__).parents[2] / "docs" / "Abnormal-Thyroid-profile-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
        document = extract_document(path)
        store = build_evidence_store(document)
        self.assertIn("T3, TOTAL | 350.00 | High | 80.00 - 200.00 | ng/dL", document.pages[0].text)
        self.assertIn("TSH | 0.20 | Low | 0.40 - 4.00 | mU/L", document.pages[0].text)
        self.assertNotIn("Smart Pathology Laboratory", document.pages[0].text)
        measurements = {item.test_name: item for item in store.measurements}
        self.assertEqual(
            set(measurements),
            {"T3, TOTAL", "T4, TOTAL", "TSH"},
        )
        self.assertEqual(
            (measurements["T3, TOTAL"].value, measurements["T3, TOTAL"].unit, measurements["T3, TOTAL"].reported_flag),
            (350, "ng/dL", "High"),
        )
        self.assertEqual(
            (measurements["T4, TOTAL"].value, measurements["T4, TOTAL"].unit, measurements["T4, TOTAL"].reported_flag),
            (28.5, "mcg/dL", "High"),
        )
        self.assertEqual(
            (measurements["TSH"].value, measurements["TSH"].unit, measurements["TSH"].reported_flag),
            (0.2, "mU/L", "Low"),
        )
