"""Unit tests for the agent tracer.

Verifies trace step creation, JSON serialization, and file persistence.
"""
import json
import pytest

from src.agentic.tracer import AgentTrace, TraceStep

pytestmark = pytest.mark.unit


class TestTraceStep:

    def test_minimal_step_to_dict(self):
        step = TraceStep(step="translation")
        d = step.to_dict()
        assert d["step"] == "translation"
        assert "result" not in d      # None values are omitted
        assert "latency_ms" not in d

    def test_step_with_result(self):
        step = TraceStep(step="language_detection", result="es")
        d = step.to_dict()
        assert d["result"] == "es"

    def test_latency_rounded_to_one_decimal(self):
        step = TraceStep(step="translation", latency_ms=123.456789)
        assert step.to_dict()["latency_ms"] == pytest.approx(123.5, abs=0.1)

    def test_model_field_present_when_set(self):
        step = TraceStep(step="expansion", model="phi-3-mini")
        assert step.to_dict()["model"] == "phi-3-mini"

    def test_extra_fields_merged(self):
        step = TraceStep(step="faiss", extra={"pool_size": 50})
        assert step.to_dict()["pool_size"] == 50


class TestAgentTrace:

    def test_trace_created_with_required_fields(self):
        trace = AgentTrace(query_id="q1", original_query="perro")
        assert trace.query_id == "q1"
        assert trace.original_query == "perro"
        assert trace.steps == []
        assert trace.final_results == []

    def test_add_step_appends(self):
        trace = AgentTrace(query_id="q1", original_query="test")
        trace.add_step(TraceStep(step="translation", result="dog"))
        trace.add_step(TraceStep(step="negation_detection", result={"detected": False}))
        assert len(trace.steps) == 2

    def test_to_dict_structure(self):
        trace = AgentTrace(query_id="q2", original_query="gato")
        trace.add_step(TraceStep(step="language_detection", result="es"))
        trace.final_results = ["2007_000001"]

        d = trace.to_dict()
        assert d["query_id"] == "q2"
        assert d["original_query"] == "gato"
        assert len(d["steps"]) == 1
        assert d["final_results"] == ["2007_000001"]
        assert "timestamp" in d

    def test_to_json_produces_valid_json(self):
        trace = AgentTrace(query_id="q3", original_query="auto")
        trace.add_step(TraceStep(step="translation", result="car", latency_ms=45.0))
        raw = trace.to_json()
        parsed = json.loads(raw)
        assert parsed["query_id"] == "q3"

    def test_to_json_is_utf8_safe(self):
        trace = AgentTrace(query_id="q4", original_query="pájaro")
        raw = trace.to_json()
        assert "pájaro" in raw

    def test_append_to_file_writes_valid_jsonl(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        t1 = AgentTrace(query_id="q1", original_query="perro")
        t2 = AgentTrace(query_id="q2", original_query="gato")
        t1.append_to_file(path)
        t2.append_to_file(path)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["query_id"] == "q1"
        assert json.loads(lines[1])["query_id"] == "q2"

    def test_load_all_from_file(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        for i in range(3):
            t = AgentTrace(query_id=f"q{i}", original_query=f"query {i}")
            t.append_to_file(path)

        loaded = AgentTrace.load_all(path)
        assert len(loaded) == 3
        assert loaded[1]["query_id"] == "q1"

    def test_load_all_returns_empty_for_missing_file(self, tmp_path):
        assert AgentTrace.load_all(tmp_path / "nonexistent.jsonl") == []
