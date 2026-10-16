from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    purpose: str
    estimated_ram_gb: float
    enabled_by_default: bool = False


PROFILES = (
    ModelProfile("BAAI/bge-small-en-v1.5", "semantic retrieval", 0.5, True),
    ModelProfile("google/flan-t5-base", "optional generation", 2.0),
    ModelProfile("facebook/bart-large-cnn", "optional summarization", 3.0),
)

