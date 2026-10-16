from __future__ import annotations

import hashlib
import os
import re

import numpy as np


class EmbeddingService:
    """Lazy local embeddings with deterministic hashed fallback when models are unavailable."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, enable_model: bool | None = None) -> None:
        self.model_name, self.dimension, self._model = model_name, dimension, None
        self.enable_model = enable_model if enable_model is not None else os.getenv("CHASM_ENABLE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        self.warning: str | None = None

    def _load(self) -> object | None:
        if self._model is not None:
            return self._model
        if not self.enable_model:
            self.warning = "Local embedding model is disabled; using deterministic hashed retrieval fallback. Set CHASM_ENABLE_LOCAL_EMBEDDINGS=true after downloading the model locally."
            return None
        try:
            from sentence_transformers import SentenceTransformer
            # Default is offline/local-first. Users may pre-download a model; the app
            # must not send a medical-document workflow to a remote service.
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except (ImportError, OSError, RuntimeError) as exc:
            self.warning = f"Semantic model unavailable; using local hashed retrieval fallback ({exc})."
        return self._model

    def _hashed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in re.findall(r"[\w.-]+", text.casefold()):
                digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
                slot = int.from_bytes(digest, "little") % self.dimension
                matrix[row, slot] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if model is None:
            return self._hashed(texts)
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

