from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chasm_rag.ingestion.document_loader import extract_document
from chasm_rag.ingestion.ocr_engine import OcrResult


class IngestionTests(unittest.TestCase):
    def test_txt_preserves_standalone_decimal_line(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("RESULTS\n123.45\nSodium: 140 mmol/L Reference Range: 135-145")
            path = Path(handle.name)
        try:
            document = extract_document(path)
            self.assertIn("123.45", document.pages[0].text)
        finally:
            path.unlink(missing_ok=True)

    def test_pdf_extraction_preserves_page_and_measurement_line(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            path = Path(handle.name)
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "LAB RESULTS\nGlucose: 110 mg/dL Reference Range: 70-99 HIGH")
        document.save(path)
        document.close()
        try:
            extracted = extract_document(path)
            self.assertEqual(extracted.pages[0].page, 1)
            self.assertIn("Glucose: 110", extracted.pages[0].text)
        finally:
            path.unlink(missing_ok=True)

    def test_pdf_uses_ocr_when_native_text_is_insufficient(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            path = Path(handle.name)
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "QR")
        pdf.save(path)
        pdf.close()
        try:
            with patch(
                "chasm_rag.ingestion.pdf_extractor.ocr_image",
                return_value=OcrResult("Scanned report content", 0.9, "OCR requires review."),
            ) as ocr:
                extracted = extract_document(path)
            ocr.assert_called_once()
            self.assertEqual(extracted.pages[0].method, "ocr")
            self.assertEqual(extracted.pages[0].text, "Scanned report content")
            self.assertIn("OCR requires review.", extracted.pages[0].warnings)
        finally:
            path.unlink(missing_ok=True)
