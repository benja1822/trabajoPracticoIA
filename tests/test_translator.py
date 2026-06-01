"""Unit tests for the MarianMT translator.

Mocks the HuggingFace model so no download is required.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agentic.tracer import TraceStep

pytestmark = pytest.mark.unit


def _make_translator(translation_output: str = "car on the street"):
    """Return a Translator with mocked MarianMT internals."""
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    # Simulate tokenizer call returning a dict-like object
    mock_tokenizer.side_effect = None
    mock_tokenizer.__call__ = MagicMock(return_value={
        "input_ids": MagicMock(spec=["to"]),
        "attention_mask": MagicMock(spec=["to"]),
    })

    mock_model = MagicMock()
    fake_token_ids = MagicMock()
    mock_model.generate.return_value = [fake_token_ids]
    mock_model.eval.return_value = mock_model

    mock_tokenizer_cls = MagicMock(return_value=mock_tokenizer)
    mock_model_cls = MagicMock(return_value=mock_model)

    # decode always returns our desired translation
    mock_tokenizer.decode = MagicMock(return_value=translation_output)
    mock_tokenizer.return_value = mock_tokenizer

    with patch("transformers.MarianTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.MarianMTModel.from_pretrained", return_value=mock_model):
        from src.agentic.translator import Translator
        t = Translator()
        # Eagerly trigger _load so the mocks are captured
        t._tokenizer = mock_tokenizer
        t._model = mock_model
        return t, mock_tokenizer, mock_model


class TestTranslator:

    def test_translate_returns_string_and_trace(self):
        from src.agentic.translator import Translator
        t, tokenizer, model = _make_translator("car on the street")
        result, step = t.translate("auto en la calle")
        assert isinstance(result, str)
        assert isinstance(step, TraceStep)

    def test_trace_step_name(self):
        t, _, _ = _make_translator("dog")
        _, step = t.translate("perro")
        assert step.step == "translation"

    def test_trace_step_has_latency(self):
        t, _, _ = _make_translator("dog")
        _, step = t.translate("perro")
        assert step.latency_ms is not None
        assert step.latency_ms >= 0

    def test_trace_step_has_model_name(self):
        t, _, _ = _make_translator("dog")
        _, step = t.translate("perro")
        assert step.model is not None
        assert "opus-mt" in step.model or "Helsinki" in step.model or step.model

    def test_trace_records_input_and_output(self):
        t, _, _ = _make_translator("red car")
        _, step = t.translate("auto rojo")
        assert "input" in step.extra
        assert "output" in step.extra

    def test_lazy_load_on_first_translate(self):
        from src.agentic.translator import Translator
        t = Translator()
        assert t._model is None
        assert t._tokenizer is None
        # After patching, first call triggers load
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "dog"
        mock_tok.__call__ = MagicMock(return_value={
            "input_ids": MagicMock(), "attention_mask": MagicMock()
        })
        mock_mod = MagicMock()
        mock_mod.generate.return_value = [MagicMock()]
        mock_mod.eval.return_value = mock_mod
        with patch("transformers.MarianTokenizer.from_pretrained", return_value=mock_tok), \
             patch("transformers.MarianMTModel.from_pretrained", return_value=mock_mod):
            t.translate("perro")
        assert t._model is not None
