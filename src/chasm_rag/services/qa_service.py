from __future__ import annotations

from ..qa.rag_qa import AnswerGenerator, GroundedAnswer, RAGQuestionAnswerer
from ..verification.claim_verifier import ClaimVerifier
from .document_service import ProcessedDocument


def answer_question(
    document: ProcessedDocument,
    question: str,
    answer_generator: AnswerGenerator | None = None,
) -> GroundedAnswer:
    if not question.strip():
        raise ValueError("Enter a question about the uploaded document.")
    return RAGQuestionAnswerer(
        document.retriever,
        ClaimVerifier(document.retriever),
        answer_generator=answer_generator,
        selected_document_id=document.store.document_id,
        valid_pages={page.page for page in document.extracted.pages},
        selected_document_name=document.extracted.source_filename,
    ).answer(question)
