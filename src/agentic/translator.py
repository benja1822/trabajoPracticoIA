import time
from typing import Optional

from src.config import TRANSLATION_MODEL
from src.agentic.tracer import TraceStep


class Translator:
    """Wraps Helsinki-NLP/opus-mt-es-en for deterministic ES→EN translation.

    Model is loaded lazily on first call.
    """

    def __init__(self, model_name: str = TRANSLATION_MODEL):
        self._model_name = model_name
        self._tokenizer = None
        self._model     = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import MarianMTModel, MarianTokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
            self._model     = MarianMTModel.from_pretrained(self._model_name)
            self._model.eval()

    def translate(self, text: str) -> tuple[str, TraceStep]:
        import torch
        self._load()
        t0 = time.perf_counter()

        inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            tokens = self._model.generate(**inputs)
        translated = self._tokenizer.decode(tokens[0], skip_special_tokens=True)

        latency = (time.perf_counter() - t0) * 1000
        step = TraceStep(
            step="translation",
            model=self._model_name,
            result=translated,
            latency_ms=latency,
            extra={"input": text, "output": translated},
        )
        return translated, step
