from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    warning: str | None = None


def ocr_image(image: object) -> OcrResult:
    """Run local Tesseract when installed; callers must retain its uncertainty."""
    try:
        import pytesseract
    except ImportError:
        return OcrResult("", None, "OCR unavailable: install pytesseract and Tesseract.")
    try:
        text = pytesseract.image_to_string(image)
        return OcrResult(text, None, "OCR text requires review; confidence is not factual validation.")
    except (OSError, RuntimeError) as exc:
        return OcrResult("", None, f"OCR failed: {exc}")

