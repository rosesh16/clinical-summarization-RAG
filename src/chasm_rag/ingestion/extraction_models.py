from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class SourceLocation:
    page: int
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ExtractedTable:
    page: int
    rows: list[list[str]]
    name: str | None = None
    confidence: float | None = None


@dataclass
class ExtractedPage:
    page: int
    text: str
    method: Literal["text", "ocr", "mixed", "none"]
    locations: list[SourceLocation] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    source_path: Path
    source_filename: str
    pages: list[ExtractedPage]
    document_id: str = field(default_factory=lambda: str(uuid4()))
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

