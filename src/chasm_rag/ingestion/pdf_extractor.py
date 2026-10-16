from __future__ import annotations

from statistics import median
from pathlib import Path

from .extraction_models import ExtractedDocument, ExtractedPage, SourceLocation
from .ocr_engine import ocr_image
from .table_extractor import tables_from_pdf_page, tables_from_text
from ..utils.exceptions import ExtractionError


def _layout_text(page: object) -> str:
    data = getattr(page, "get_text")("dict", sort=True)
    spans: list[tuple[float, float, float, str, tuple[float, float, float, float]]] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if not text:
                    continue
                bbox = tuple(float(value) for value in span["bbox"])
                baseline = float(span.get("origin", (0, bbox[3]))[1])
                spans.append((baseline, bbox[0], float(span.get("size", 0)), text, bbox))
    if not spans:
        return ""

    positive_sizes = [size for _, _, size, _, _ in spans if size > 0]
    typical_size = median(positive_sizes) if positive_sizes else 10.0
    body_spans = [span for span in spans if span[2] <= typical_size * 2.2]

    def overlaps(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
        width = min(first[2], second[2]) - max(first[0], second[0])
        height = min(first[3], second[3]) - max(first[1], second[1])
        return width > 0 and height > 0

    # Large text painted over normal content is commonly a watermark, not a
    # second copy of the underlying words.
    spans = [
        span for span in spans
        if span[2] <= typical_size * 2.2
        or not any(overlaps(span[4], body[4]) for body in body_spans)
    ]

    visual_lines: list[list[tuple[float, float, float, str, tuple[float, float, float, float]]]] = []
    for span in sorted(spans, key=lambda item: (item[0], item[1])):
        if visual_lines and abs(span[0] - visual_lines[-1][0][0]) <= 2.5:
            visual_lines[-1].append(span)
        else:
            visual_lines.append([span])

    output: list[str] = []
    previous_baseline: float | None = None
    for visual_line in visual_lines:
        baseline = median(span[0] for span in visual_line)
        if previous_baseline is not None and baseline - previous_baseline > typical_size * 2.2:
            output.append("")
        ordered_spans = sorted(visual_line, key=lambda item: item[1])
        line_parts = [ordered_spans[0][3]]
        previous_right = ordered_spans[0][4][2]
        for span in ordered_spans[1:]:
            gap = span[1] - previous_right
            separator = " | " if gap > typical_size * 1.3 else " "
            line_parts.extend((separator, span[3]))
            previous_right = span[4][2]
        output.append("".join(line_parts))
        previous_baseline = baseline
    return "\n".join(output)


def extract_pdf(path: Path, use_ocr: bool = True) -> ExtractedDocument:
    try:
        import fitz
    except ImportError as exc:
        raise ExtractionError("PDF support requires PyMuPDF. Run: pip install -r requirements.txt") from exc
    try:
        pdf = fitz.open(path)
    except (RuntimeError, OSError) as exc:
        raise ExtractionError(f"Could not open PDF: {exc}") from exc
    if pdf.needs_pass:
        raise ExtractionError("Password-protected PDFs are not supported.")
    pages: list[ExtractedPage] = []
    for number, page in enumerate(pdf, start=1):
        blocks = page.get_text("blocks", sort=True)
        text = _layout_text(page)
        locations = [SourceLocation(number, bbox=tuple(block[:4])) for block in blocks if block[4].strip()]
        warnings: list[str] = []
        method = "text"
        confidence: float | None = 1.0 if text else None
        useful_text = sum(character.isalnum() for character in text)
        if use_ocr and (not text.strip() or useful_text < 30):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            result = ocr_image(pix.pil_image())
            if result.text.strip():
                text, confidence, method = result.text, result.confidence, "ocr"
            if result.warning:
                warnings.append(result.warning)
        if not text:
            method = "none"
            warnings.append("No extractable text on this page; human review may be required.")
        tables = tables_from_pdf_page(number, page) or tables_from_text(number, text)
        pages.append(ExtractedPage(number, text, method, locations, tables, confidence, warnings))
    document = ExtractedDocument(path, path.name, pages)
    if not document.text.strip():
        document.warnings.append("Document contained no extractable text.")
    return document
