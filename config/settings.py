from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Settings intentionally kept small and overridable through environment variables."""

    embedding_model: str = os.getenv("CHASM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    generator_mode: str = os.getenv("CHASM_GENERATOR_MODE", "extractive")
    generator_model: str = os.getenv("CHASM_GENERATOR_MODEL", "google/flan-t5-base")
    chunk_tokens: int = int(os.getenv("CHASM_CHUNK_TOKENS", "350"))
    chunk_overlap: int = int(os.getenv("CHASM_CHUNK_OVERLAP", "40"))
    use_gpu: bool = os.getenv("CHASM_USE_GPU", "false").lower() == "true"
    artifact_dir: Path = Path(os.getenv("CHASM_ARTIFACT_DIR", "artifacts"))


settings = Settings()

