"""Unit tests for the semantic query expander.

LLMClient is mocked — no model download required.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agentic.tracer import TraceStep

pytestmark = pytest.mark.unit

VALID_EXPANSION_JSON = '{"expanded": "dog canine pet animal", "variants": ["dog playing fetch", "puppy running"]}'
INVALID_JSON         = "Here is the expansion: dog canine (no json here)"


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.model_name = "phi-3-mini-mock"
    with patch("src.agentic.expander.LLMClient") as MockClass:
        MockClass.get.return_value = client
        yield client


class TestExpandQuery:

    def test_returns_expanded_string_and_variants(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_EXPANSION_JSON, 120.0)
        mock_llm_client.extract_json.return_value = {
            "expanded": "dog canine pet animal",
            "variants": ["dog playing fetch", "puppy running"],
        }
        from src.agentic.expander import expand_query
        expanded, variants, step = expand_query("dog")
        assert expanded == "dog canine pet animal"
        assert len(variants) == 2

    def test_returns_trace_step(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_EXPANSION_JSON, 80.0)
        mock_llm_client.extract_json.return_value = {"expanded": "dog", "variants": []}
        from src.agentic.expander import expand_query
        _, _, step = expand_query("dog")
        assert isinstance(step, TraceStep)
        assert step.step == "semantic_expansion"

    def test_trace_has_latency(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_EXPANSION_JSON, 250.0)
        mock_llm_client.extract_json.return_value = {"expanded": "dog", "variants": []}
        from src.agentic.expander import expand_query
        _, _, step = expand_query("dog")
        assert step.latency_ms == pytest.approx(250.0)

    def test_fallback_on_invalid_json(self, mock_llm_client):
        mock_llm_client.generate.return_value = (INVALID_JSON, 90.0)
        mock_llm_client.extract_json.side_effect = ValueError("No JSON found")
        from src.agentic.expander import expand_query
        expanded, variants, step = expand_query("dog")
        assert expanded == "dog"    # original query preserved
        assert variants == []

    def test_fallback_on_llm_exception(self, mock_llm_client):
        mock_llm_client.generate.side_effect = RuntimeError("Model failed")
        from src.agentic.expander import expand_query
        expanded, variants, step = expand_query("cat")
        assert expanded == "cat"
        assert variants == []

    def test_variants_empty_list_when_not_provided(self, mock_llm_client):
        mock_llm_client.generate.return_value = ('{"expanded": "car vehicle"}', 50.0)
        mock_llm_client.extract_json.return_value = {"expanded": "car vehicle"}  # no variants key
        from src.agentic.expander import expand_query
        _, variants, _ = expand_query("car")
        assert variants == []
