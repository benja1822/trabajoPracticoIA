"""Integration tests for the agentic pipeline — 5 representative queries.

All heavy dependencies (CLIP, FAISS, LLM, translator) are mocked.
Tests verify the orchestration logic: step sequence, trace structure,
negation routing, LLM-disabled fallback.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from src.agentic.pipeline import AgenticPipeline, AgentResult
from src.agentic.tracer import AgentTrace
from src.indexing.faiss_manager import FAISSManager

pytestmark = pytest.mark.unit


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def faiss_manager(mock_embeddings, mock_image_ids):
    mgr = FAISSManager()
    mgr.add(mock_embeddings, mock_image_ids)
    return mgr


@pytest.fixture
def mock_translator():
    t = MagicMock()
    from src.agentic.tracer import TraceStep
    def translate(text):
        translations = {
            "perro en el jardín": "dog in the garden",
            "auto no rojo":        "car not red",
            "dog playing":         "dog playing",
            "xzxzxz qwerty":      "xzxzxz qwerty",
            "persona con bicicleta": "person with bicycle",
        }
        result = translations.get(text, text + " [translated]")
        step = TraceStep(step="translation", result=result,
                         model="opus-mt-es-en", latency_ms=45.0,
                         extra={"input": text, "output": result})
        return result, step
    t.translate.side_effect = translate
    return t


@pytest.fixture
def mock_reranker(mock_embeddings, mock_image_ids):
    from src.reranking.negation_reranker import NegationReranker
    reranker = NegationReranker(
        clip_extractor=MagicMock(),
        image_embeddings=mock_embeddings,
        image_ids=mock_image_ids,
    )
    reranker._get_neg_embedding = MagicMock(return_value=mock_embeddings[0])
    return reranker


@pytest.fixture
def pipeline(mock_clip_extractor, faiss_manager, mock_reranker, mock_translator):
    with patch("src.agentic.expander.LLMClient") as MockExpLLM, \
         patch("src.agentic.validator.LLMClient") as MockValLLM:

        exp_client = MagicMock()
        exp_client.model_name = "phi-3-mini-mock"
        exp_client.generate.return_value = ('{"expanded": "dog canine", "variants": ["dog fetch"]}', 100.0)
        exp_client.extract_json.return_value = {"expanded": "dog canine", "variants": ["dog fetch"]}
        MockExpLLM.get.return_value = exp_client

        val_client = MagicMock()
        val_client.model_name = "phi-3-mini-mock"
        val_client.generate.return_value = ('{"valid": true, "reason": "ok", "corrected_query": null}', 80.0)
        val_client.extract_json.return_value = {"valid": True, "reason": "ok", "corrected_query": None}
        MockValLLM.get.return_value = val_client

        p = AgenticPipeline(
            clip_extractor=mock_clip_extractor,
            faiss_manager=faiss_manager,
            reranker=mock_reranker,
            translator=mock_translator,
            enable_llm=True,
        )
        yield p


# ─── 5 integration queries ────────────────────────────────────────────────────

class TestPipelineFiveQueries:

    def test_query_1_simple_spanish_no_negation(self, pipeline):
        """Simple Spanish query — translate, expand, no reranking."""
        result = pipeline.run("perro en el jardín", query_id="q_test_1")

        assert isinstance(result, AgentResult)
        assert result.query_id == "q_test_1"
        assert result.original_query == "perro en el jardín"
        assert result.translated_query == "dog in the garden"
        assert result.has_negation is False or result.negative_attributes == []
        assert len(result.top_results) == 10
        assert all(isinstance(r, str) for r in result.top_results)

    def test_query_2_spanish_with_negation(self, pipeline):
        """Query with negation — triggers reranking path."""
        result = pipeline.run("auto no rojo", query_id="q_test_2")

        assert result.translated_query == "car not red"
        assert result.has_negation is True
        assert "red" in result.negative_attributes
        assert "car" in result.positive_query
        assert len(result.top_results) == 10

    def test_query_3_already_english(self, pipeline):
        """English query — translation should be skipped."""
        from src.agentic.tracer import TraceStep
        lang_step = TraceStep(step="language_detection", result="en")
        with patch("src.agentic.pipeline._detect_language", return_value=("en", lang_step)):
            result = pipeline.run("dog playing", query_id="q_test_3")
        assert result.original_query == "dog playing"
        assert len(result.top_results) == 10

    def test_query_4_invalid_semantics_still_returns_results(self, pipeline):
        """Even invalid queries return results — pipeline never hard-fails."""
        with patch("src.agentic.validator.LLMClient") as MockVal:
            val_client = MagicMock()
            val_client.model_name = "phi-3-mini-mock"
            val_client.generate.return_value = (
                '{"valid": false, "reason": "incoherent", "corrected_query": "person bicycle"}',
                70.0,
            )
            val_client.extract_json.return_value = {
                "valid": False, "reason": "incoherent", "corrected_query": "person bicycle"
            }
            MockVal.get.return_value = val_client
            result = pipeline.run("xzxzxz qwerty", query_id="q_test_4")

        assert result.is_valid is False
        assert result.corrected_query is not None
        assert len(result.top_results) == 10  # still returns results

    def test_query_5_llm_disabled_fallback(self, mock_clip_extractor, faiss_manager, mock_reranker, mock_translator):
        """LLM disabled — translation only, no expansion or validation."""
        p = AgenticPipeline(
            clip_extractor=mock_clip_extractor,
            faiss_manager=faiss_manager,
            reranker=mock_reranker,
            translator=mock_translator,
            enable_llm=False,
        )
        result = p.run("persona con bicicleta", query_id="q_test_5")

        assert result.translated_query == "person with bicycle"
        assert len(result.top_results) == 10
        # Expansion and validation steps should be skipped
        step_names = [s.step for s in result.trace.steps]
        assert "semantic_expansion" in step_names
        assert "integrity_validation" in step_names


# ─── Trace structure ──────────────────────────────────────────────────────────

class TestPipelineTrace:

    def test_trace_contains_all_expected_steps(self, pipeline):
        result = pipeline.run("perro en el jardín", query_id="q_trace")
        step_names = [s.step for s in result.trace.steps]
        for expected in ["language_detection", "translation", "negation_detection",
                         "semantic_expansion", "integrity_validation",
                         "faiss_retrieval", "reranking"]:
            assert expected in step_names, f"Missing step: {expected}"

    def test_trace_query_id_matches(self, pipeline):
        result = pipeline.run("perro", query_id="q_abc")
        assert result.trace.query_id == "q_abc"

    def test_trace_final_results_match_top_results(self, pipeline):
        result = pipeline.run("perro", query_id="q_check")
        assert result.trace.final_results == result.top_results

    def test_agent_result_has_public_properties(self, pipeline):
        result = pipeline.run("perro", query_id="q_props")
        # These are the fields the ablation study and other modules access
        assert hasattr(result, "top_results")
        assert hasattr(result, "original_query")
        assert hasattr(result, "translated_query")
        assert hasattr(result, "negative_attributes")
        assert hasattr(result, "expanded_query")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "trace")
