"""Unit tests for LLMClient.

Tests the static JSON extraction utility and singleton pattern
without loading any real model.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.agentic.llm_client import LLMClient

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_singleton():
    """Isolate singleton state between tests."""
    original = LLMClient._instance
    LLMClient._instance = None
    yield
    LLMClient._instance = original


class TestExtractJson:

    def test_clean_json_object(self):
        raw = '{"expanded": "dog playing", "variants": ["dog fetch", "dog run"]}'
        result = LLMClient.extract_json(raw)
        assert result["expanded"] == "dog playing"
        assert len(result["variants"]) == 2

    def test_strips_markdown_fence(self):
        raw = '```json\n{"valid": true, "reason": "ok"}\n```'
        result = LLMClient.extract_json(raw)
        assert result["valid"] is True

    def test_strips_plain_fence(self):
        raw = '```\n{"valid": false}\n```'
        result = LLMClient.extract_json(raw)
        assert result["valid"] is False

    def test_extracts_json_with_surrounding_text(self):
        raw = 'Here is the result: {"key": "value"} — done.'
        result = LLMClient.extract_json(raw)
        assert result["key"] == "value"

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            LLMClient.extract_json("this is just plain text without braces")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            LLMClient.extract_json("")

    def test_nested_json(self):
        raw = '{"result": {"nested": true}, "count": 3}'
        result = LLMClient.extract_json(raw)
        assert result["result"]["nested"] is True
        assert result["count"] == 3

    @pytest.mark.parametrize("text,expected_key,expected_val", [
        ('{"valid": true, "reason": "coherent"}', "valid", True),
        ('{"valid": false, "reason": "contradiction"}', "valid", False),
        ('{"expanded": "car vehicle", "variants": []}', "expanded", "car vehicle"),
    ])
    def test_parametrized_json_shapes(self, text, expected_key, expected_val):
        result = LLMClient.extract_json(text)
        assert result[expected_key] == expected_val


class TestSingleton:

    def test_get_returns_same_instance(self):
        a = LLMClient.get()
        b = LLMClient.get()
        assert a is b

    def test_get_creates_instance_if_none(self):
        assert LLMClient._instance is None
        instance = LLMClient.get()
        assert instance is not None
        assert LLMClient._instance is instance

    def test_model_name_none_before_load(self):
        client = LLMClient.get()
        assert client.model_name is None
