# Architecture

```text
PDF/DOCX/TXT → extraction/OCR warning → canonical evidence store
                                      ├→ structured measurements + validation
                                      └→ BM25 + local embeddings → hybrid retrieval
evidence → tokenizer-aware chunks → CHASM draft → claim verifier → safety gate → UI
```

Every evidence record retains its source text, page, type, source location where available, extraction confidence, and warnings. Document services construct a retriever only from the selected document, preventing cross-document evidence mixing.

