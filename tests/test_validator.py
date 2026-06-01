"""Unit tests for the semantic integrity validator.

LLMClient is mocked — no model download required.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agentic.tracer import TraceStep

pytestmark = pytest.mark.unit

VALID_RESPONSE   = '{"valid": true,  "reason": "query is coherent", "corrected_query": null}'
INVALID_RESPONSE = '{"valid": false, "reason": "car cannot be both red and not red", "corrected_query": "car on the street"}'


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.model_name = "phi-3-mini-mock"
    with patch("src.agentic.validator.LLMClient") as MockClass:
        MockClass.get.return_value = client
        yield client


class TestValidateQuery:

    def test_valid_query_returns_true(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_RESPONSE, 95.0)
        mock_llm_client.extract_json.return_value = {
            "valid": True, "reason": "query is coherent", "corrected_query": None
        }
        from src.agentic.validator import validate_query
        is_valid, reason, corrected, step = validate_query("car on the street", [])
        assert is_valid is True
        assert corrected is None

    def test_invalid_query_returns_false_with_reason(self, mock_llm_client):
        mock_llm_client.generate.return_value = (INVALID_RESPONSE, 110.0)
        mock_llm_client.extract_json.return_value = {
            "valid": False,
            "reason": "car cannot be both red and not red",
            "corrected_query": "car on the street",
        }
        from src.agentic.validator import validate_query
        is_valid, reason, corrected, step = validate_query("car red and not red", ["red"])
        assert is_valid is False
        assert "red" in reason.lower() or "cannot" in reason.lower()
        assert corrected == "car on the street"

    def test_returns_trace_step(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_RESPONSE, 80.0)
        mock_llm_client.extract_json.return_value = {"valid": True, "reason": "ok", "corrected_query": None}
        from src.agentic.validator import validate_query
        _, _, _, step = validate_query("dog playing", [])
        assert isinstance(step, TraceStep)
        assert step.step == "integrity_validation"

    def test_fallback_on_llm_exception_defaults_to_valid(self, mock_llm_client):
        mock_llm_client.generate.side_effect = RuntimeError("LLM unavailable")
        from src.agentic.validator import validate_query
        is_valid, reason, corrected, step = validate_query("dog playing", [])
        assert is_valid is True      # safe default: don't block the pipeline
        assert corrected is None

    def test_fallback_on_invalid_json_defaults_to_valid(self, mock_llm_client):
        mock_llm_client.generate.return_value = ("not json at all", 60.0)
        mock_llm_client.extract_json.side_effect = ValueError("No JSON")
        from src.agentic.validator import validate_query
        is_valid, _, _, _ = validate_query("bird flying", ["ground"])
        assert is_valid is True

    def test_corrected_query_none_when_not_provided(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_RESPONSE, 70.0)
        mock_llm_client.extract_json.return_value = {"valid": True, "reason": "ok"}
        from src.agentic.validator import validate_query
        _, _, corrected, _ = validate_query("cat sleeping", [])
        assert corrected is None

    def test_trace_has_valid_result_field(self, mock_llm_client):
        mock_llm_client.generate.return_value = (VALID_RESPONSE, 90.0)
        mock_llm_client.extract_json.return_value = {"valid": True, "reason": "ok", "corrected_query": None}
        from src.agentic.validator import validate_query
        _, _, _, step = validate_query("bird", [])
        assert "valid" in step.result
        assert "reason" in step.result
