"""Evaluation metric definitions.

Unsupported-claim rate = unsupported or contradicted generated claims / all claims.
Evidence coverage = claims with at least one correct source citation / all claims.
Retrieval Recall@k = relevant evidence returned in top k / all annotated relevant evidence.
Numerical accuracy = exact supported values and units / assessed source values.
ROUGE-L/BERTScore are optional complementary text metrics, not safety measures.
"""

