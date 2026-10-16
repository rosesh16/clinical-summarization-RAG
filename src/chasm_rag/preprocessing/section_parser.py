from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Section:
    title: str
    text: str
    start: int
    end: int


_HEADING = re.compile(r"^(?:[A-Z][A-Z /&-]{2,}|[A-Z][\w /&-]{2,}):$", re.MULTILINE)


def parse_sections(text: str) -> list[Section]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [Section("Unsectioned", text, 0, len(text))] if text.strip() else []
    sections: list[Section] = []
    if matches[0].start() > 0:
        sections.append(Section("Unsectioned", text[:matches[0].start()].strip(), 0, matches[0].start()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        sections.append(Section(match.group().rstrip(":"), body, match.start(), end))
    return [section for section in sections if section.text]

