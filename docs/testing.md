# Testing

Run `python -m unittest discover -s tests -v` from the repository root after `pip install -e .`. Fixtures contain no patient data. Coverage includes TXT/PDF extraction, decimal preservation, ranges and conflicting measurements, chunks and evidence IDs, hybrid retrieval, end-to-end QA, and a static generator that deliberately emits an unsupported result so fallback behavior is deterministic rather than relying on a model hallucinating.

