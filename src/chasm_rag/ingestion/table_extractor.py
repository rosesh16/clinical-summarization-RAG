from __future__ import annotations

from .extraction_models import ExtractedTable


def tables_from_text(page: int, text: str) -> list[ExtractedTable]:
    """Conservative delimiter-table preservation, never converts cells into prose."""
    rows = [[cell.strip() for cell in line.split("|")] for line in text.splitlines() if line.count("|") >= 2]
    return [ExtractedTable(page=page, rows=rows, name="Delimited table")] if rows else []


def tables_from_pdf_page(page_number: int, pdf_page: object) -> list[ExtractedTable]:
    """Use PyMuPDF's layout table detector when available, retaining cells as rows."""
    try:
        found = pdf_page.find_tables()
    except (AttributeError, RuntimeError, ValueError):
        return []
    tables: list[ExtractedTable] = []
    for index, table in enumerate(getattr(found, "tables", []), start=1):
        rows = [["" if cell is None else str(cell) for cell in row] for row in table.extract()]
        if rows:
            tables.append(ExtractedTable(page=page_number, rows=rows, name=f"PDF table {index}"))
    return tables

