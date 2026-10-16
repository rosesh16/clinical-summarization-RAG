from pathlib import Path
import unittest

from chasm_rag.services.document_service import process_document


class RetrievalTests(unittest.TestCase):
    def test_exact_value_is_retrieved_from_document_local_index(self) -> None:
        doc = process_document(Path(__file__).parents[1] / "fixtures" / "lab_report.txt")
        results = doc.retriever.search("What is LDL cholesterol 165 mg/dL?")
        self.assertTrue(any("LDL Cholesterol" in item.evidence.original_text for item in results))

