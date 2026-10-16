from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from chasm_rag.services.document_service import process_document
from chasm_rag.services.qa_service import answer_question


def main() -> None:
    fixture = Path(__file__).parents[1] / "tests" / "fixtures" / "lab_report.txt"
    document = process_document(fixture)
    answer = answer_question(document, "What is the LDL cholesterol value?")
    metric = {"fixture": fixture.name, "retrieval_recall_at_6_for_ldl": any("LDL Cholesterol" in result.evidence.original_text for result in answer.evidence), "evidence_count": len(answer.evidence), "note": "This is an executable fixture evaluation, not a clinical-quality claim."}
    output = Path(__file__).parents[1] / "evaluation" / "results" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metric, indent=2), encoding="utf-8")
    print(json.dumps(metric, indent=2))


if __name__ == "__main__":
    main()
