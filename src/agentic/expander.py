from typing import Dict, List

from src.agentic.llm_client import LLMClient
from src.agentic.prompts import EXPANSION_SYSTEM, EXPANSION_USER, build_phi3_messages
from src.agentic.tracer import TraceStep
from src.config import LLM_TEMP_EXPANSION


def expand_query(positive_query: str) -> tuple[str, List[str], TraceStep]:
    """Expand a query with synonyms and generate alternative phrasings.

    Returns (expanded_query, variants, trace_step).
    On any LLM failure falls back to returning the original query unchanged.
    """
    client = LLMClient.get()
    messages = build_phi3_messages(
        EXPANSION_SYSTEM,
        EXPANSION_USER.format(query=positive_query),
    )

    try:
        raw, latency = client.generate(messages, temperature=LLM_TEMP_EXPANSION)
        parsed: Dict = client.extract_json(raw)
        expanded = parsed.get("expanded", positive_query)
        variants = parsed.get("variants", [])
        if not isinstance(variants, list):
            variants = []
    except Exception as e:
        expanded = positive_query
        variants = []
        latency  = 0.0
        raw      = f"[fallback: {e}]"

    step = TraceStep(
        step="semantic_expansion",
        model=client.model_name,
        latency_ms=latency,
        result={"expanded": expanded, "variants": variants},
        extra={"input": positive_query, "raw_output": raw[:200]},
    )
    return expanded, variants, step
