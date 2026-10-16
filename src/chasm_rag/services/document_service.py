from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..evidence.evidence_store import EvidenceStore, build_evidence_store
from ..ingestion.document_loader import extract_document
from ..ingestion.extraction_models import ExtractedDocument
from ..retrieval.hybrid_retriever import HybridRetriever


@dataclass
class ProcessedDocument:
    extracted: ExtractedDocument
    store: EvidenceStore
    retriever: HybridRetriever


def process_document(path: str | Path) -> ProcessedDocument:
    extracted = extract_document(path)
    store = build_evidence_store(extracted)
    return ProcessedDocument(extracted, store, HybridRetriever(store.evidence))

