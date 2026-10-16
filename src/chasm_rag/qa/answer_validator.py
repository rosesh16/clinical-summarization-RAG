from __future__ import annotations

from ..verification.claim_verifier import ClaimVerifier, ClaimVerification


def validate_answer(answer: str, verifier: ClaimVerifier) -> list[ClaimVerification]:
    return verifier.verify(answer)

