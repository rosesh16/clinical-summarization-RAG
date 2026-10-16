# CHASM-RAG

Chunk-Hierarchical Abstractive Summarization with Retrieval-Augmented Generation and Evidence Verification is a local-first research/education application for clinical document processing. It extracts PDF/DOCX/TXT source content, preserves measurements and ranges, retrieves evidence from the selected document, produces a hierarchical summary, verifies claims, and falls back to source-derived output when verification fails.

It does **not** diagnose, prescribe treatment, or replace a clinician.

## Architecture

See [architecture documentation](docs/architecture.md). Core modules are ingestion, preprocessing, evidence, retrieval, summarization, verification, QA, and shared services. The Streamlit UI only calls services; it does not duplicate business logic.

## Dataset, training, and evaluation

RAG does not require supervised fine-tuning: it indexes the uploaded runtime document as its knowledge base. This is distinct from the synthetic/manual **evaluation corpus**, which measures extraction, retrieval and verification but is not model training data. Fine-tuning is outside this initial scope.

## Windows setup

Use Python 3.10 or 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
streamlit run app.py
```

PyMuPDF enables PDF extraction. PDF text is reconstructed in visual reading order from positioned text spans, keeping cells on the same row and suppressing large overlapping watermark text. Pages with little selectable text use local OCR when Tesseract is available. For OCR, install the Tesseract Windows application separately, ensure `tesseract.exe` is on `PATH`, then install the listed `pytesseract` package. The app reports unavailable OCR instead of silently treating a scanned page as extracted. Complex layouts can still require review; compare the extracted-text view with the source page.

## Models and hardware

The default summarizer is extractive and requires no model download. A configurable, lazy Hugging Face seq2seq generator is available by setting `CHASM_GENERATOR_MODE=transformers`; it only loads locally cached models, falls back to extractive summarization if unavailable, and retries on CPU after CUDA out-of-memory. Semantic retrieval has an implemented lazy `BAAI/bge-small-en-v1.5` adapter, but defaults to a clearly disclosed local hashed-vector fallback so a first run cannot download a model or transmit a document. To use a pre-downloaded local embedding model, set `CHASM_ENABLE_LOCAL_EMBEDDINGS=true`; it is loaded with offline-only mode. On a 4 GB RTX 3050, use the small embedding model and CPU fallback; optional `flan-t5-base` generation may be slow or memory constrained. First model acquisition is a user-controlled download into the local Hugging Face cache; normal runs reuse that cache. Do not enable/download heavier models without measuring memory and quality.

## Tests and evaluation

```powershell
python -m unittest discover -s tests -v
python scripts/run_evaluation.py
```

The evaluation script records actual fixture results in `evaluation/results/latest.json`. Record model names, configurations, held-out documents, latency/memory and measured values before making A/B/C research claims. Definitions live in `evaluation/metrics.py`.

## Demonstration workflow

1. Upload a de-identified PDF/DOCX/TXT.
2. Inspect extraction warnings and measurement rows.
3. Generate a summary and inspect per-claim verification.
4. Ask a document-specific question and inspect source excerpts/page citations.
5. Show that an unsupported generated value triggers the conservative fallback (covered by automated test).

## Team division

- Student 1: ingestion, layout extraction, chunks, measurements.
- Student 2: retrieval, embeddings, CHASM orchestration and QA.
- Student 3: verification, safety gate, UI, evaluation and documentation.

All changes share the canonical evidence models in `src/chasm_rag/evidence/evidence_models.py`.

## Known limitations

Read [limitations](docs/limitations.md) before use with real records. Never commit patient data, model caches, temporary uploads, or generated indexes.
