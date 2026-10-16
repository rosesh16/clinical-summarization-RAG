from __future__ import annotations

import re


def normalize_for_retrieval(text: str) -> str:
    """Retrieval-only normalization; never use this in place of source text."""
    return re.sub(r"\s+", " ", text.casefold()).strip()

