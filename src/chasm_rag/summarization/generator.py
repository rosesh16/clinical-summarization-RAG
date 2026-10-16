from __future__ import annotations

import re
from typing import Protocol


class Generator(Protocol):
    def generate(self, text: str, max_new_tokens: int = 220) -> str: ...


class ExtractiveGenerator:
    """Safe default that produces source sentences without pretrained-model downloads."""
    def generate(self, text: str, max_new_tokens: int = 220) -> str:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        ranked = sorted(enumerate(sentences), key=lambda pair: (sum(char.isdigit() for char in pair[1]), len(pair[1])), reverse=True)
        selected = sorted(index for index, sentence in ranked[:5] if sentence.strip())
        output = " ".join(sentences[index].strip() for index in selected)
        return output[:max_new_tokens * 5]


class StaticGenerator:
    """Test-only controllable generator for deterministic verification tests."""
    def __init__(self, output: str) -> None:
        self.output = output
    def generate(self, text: str, max_new_tokens: int = 220) -> str:
        return self.output


class TransformersGenerator:
    """Lazy, offline-only Hugging Face seq2seq generator for locally cached models."""
    def __init__(self, model_name: str, use_gpu: bool = False) -> None:
        self.model_name, self.use_gpu = model_name, use_gpu
        self._tokenizer = self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            if self.use_gpu and torch.cuda.is_available():
                self._model.to("cuda")
            self._model.eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Local generator '{self.model_name}' is unavailable. Download it explicitly before enabling it. {exc}") from exc

    def generate(self, text: str, max_new_tokens: int = 220) -> str:
        self._load()
        import torch
        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        try:
            with torch.inference_mode():
                output = self._model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        except torch.cuda.OutOfMemoryError:
            self._model.to("cpu")
            self.use_gpu = False
            return self.generate(text, max_new_tokens)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

