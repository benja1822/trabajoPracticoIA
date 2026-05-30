"""Singleton wrapper for the lightweight LLM (Phi-3-mini or TinyLlama).

Both the expander and the validator share this instance to avoid loading
the model twice — critical on Kaggle where VRAM is finite.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from src.config import (
    DEVICE, LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK,
    LLM_MAX_NEW_TOKENS, LLM_TEMP_EXPANSION, LLM_TEMP_REASONING,
)


class LLMClient:
    _instance: Optional["LLMClient"] = None

    def __init__(self):
        self._pipeline = None
        self._model_name: Optional[str] = None

    @classmethod
    def get(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline, BitsAndBytesConfig

        use_quantization = DEVICE == "cuda"
        quantization_cfg = None
        if use_quantization:
            try:
                quantization_cfg = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                use_quantization = False

        for model_name in (LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK):
            try:
                kwargs: Dict[str, Any] = {
                    "task":             "text-generation",
                    "model":            model_name,
                    "device_map":       "auto" if DEVICE == "cuda" else None,
                    "torch_dtype":      torch.float16 if DEVICE == "cuda" else torch.float32,
                    "trust_remote_code": True,
                }
                if quantization_cfg:
                    kwargs["model_kwargs"] = {"quantization_config": quantization_cfg}
                if DEVICE == "cpu":
                    kwargs["device"] = -1

                self._pipeline  = pipeline(**kwargs)
                self._model_name = model_name
                print(f"LLM loaded: {model_name}")
                return
            except Exception as e:
                print(f"Could not load {model_name}: {e}")

        raise RuntimeError("No LLM model could be loaded. Check your environment.")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMP_REASONING,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    ) -> tuple[str, float]:
        """Returns (raw_text, latency_ms)."""
        self._load()
        t0 = time.perf_counter()

        do_sample = temperature > 0
        output = self._pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            return_full_text=False,
        )
        raw = output[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")

        latency = (time.perf_counter() - t0) * 1000
        return raw.strip(), latency

    @staticmethod
    def extract_json(text: str) -> Dict:
        """Extract first JSON object from generated text, tolerant of markdown fences."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM output: {text!r}")
        return json.loads(match.group())

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name
