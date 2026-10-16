from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str


def extract_claims(text: str) -> list[Claim]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]
    return [Claim(f"claim-{index}", piece) for index, piece in enumerate(pieces, 1)]

