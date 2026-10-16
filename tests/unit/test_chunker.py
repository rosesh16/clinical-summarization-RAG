import unittest

from chasm_rag.evidence.evidence_models import EvidenceType, EvidenceUnit
from chasm_rag.preprocessing.chunker import chunk_evidence


class ChunkingTests(unittest.TestCase):
    def test_evidence_ids_are_retained(self) -> None:
        evidence = [EvidenceUnit("Alpha measurement 1", "alpha measurement 1", "doc", 1, EvidenceType.NARRATIVE), EvidenceUnit("Beta measurement 2", "beta measurement 2", "doc", 1, EvidenceType.NARRATIVE)]
        chunks = chunk_evidence(evidence, max_tokens=5, overlap=0)
        self.assertEqual({identifier for chunk in chunks for identifier in chunk.evidence_ids}, {item.evidence_id for item in evidence})
        self.assertTrue(all(chunk.token_count <= 5 for chunk in chunks))
