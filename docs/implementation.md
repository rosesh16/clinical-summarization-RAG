# Implementation progress

- Phase 0: empty workspace assessed; Python 3.10 confirmed; no existing Git history.
- Phase 1: PDF/DOCX/TXT ingestion, optional OCR, canonical extraction and Streamlit upload implemented.
- Phase 2: section parser, tokenizer-aware chunker, structured measurements, reference ranges and conflict checks implemented.
- Phase 3: per-document BM25, local embedding adapter with deterministic fallback, vector search and reciprocal-rank fusion implemented.
- Phase 4–6: extractive CHASM baseline, claim verification, safety fallback, grounded QA and UI integration implemented.
- Phase 7: synthetic fixtures, executable tests/evaluation, architecture and safety documentation added.
- Lipid-report refinement: a dedicated column-major layout parser reconstructs lipid-panel rows without misclassifying registration dates, age, or patient identifiers. The UI separately displays only high/elevated results and source-provided follow-up text.
- Lipid safety refinement: fallback summaries use deterministic source measurements, explicitly label the total-minus-HDL arithmetic check, and retain the report's follow-up note. Lipid RAG questions return the complete structured panel as page-cited evidence.
- Conservative fallbacks now summarize thyroid measurements in prose, state the elevated-T3/elevated-T4/low-TSH pattern without diagnosing, and retain explanatory caveats only when those topics appear in the source. Other measurement reports use readable source-range comparisons rather than a raw list of values.
- Reference-range QA questions are answered from structured measurements instead of boilerplate retrieval snippets; exact source precision is retained so numeric verification can validate the generated answer.
- QA supports an injectable question-and-evidence answer generator, but no such generator is configured by default. Without one, it uses cited extractive source excerpts; every answer branch verifies against the selected document's evidence, and failed generated answers are retried once before falling back conservatively.

The default extractive generator avoids a model download. Configurable pretrained generation is intentionally a future adapter point; any such output must still pass the same verifier and gate.
