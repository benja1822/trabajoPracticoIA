import json
from typing import List, Optional

from src.agentic.llm_client import LLMClient
from src.agentic.prompts import VALIDATION_SYSTEM, VALIDATION_USER, build_phi3_messages
from src.agentic.tracer import TraceStep
from src.config import LLM_TEMP_REASONING


def validate_query(
    positive_query: str,
    negative_attributes: List[str],
) -> tuple[bool, str, Optional[str], TraceStep]:
    """Check semantic integrity of the (positive_query, negative_attributes) pair.

    Returns (is_valid, reason, corrected_query_or_None, trace_step).
    Defaults to valid=True on any LLM failure so the pipeline keeps running.
    """
    client = LLMClient.get()
    neg_str = json.dumps(negative_attributes, ensure_ascii=False)
    messages = build_phi3_messages(
        VALIDATION_SYSTEM,
        VALIDATION_USER.format(
            positive_query=positive_query,
            negative_attributes=neg_str,
        ),
    )

    is_valid       = True
    reason         = "validation skipped (LLM unavailable)"
    corrected      = None
    latency        = 0.0
    raw            = ""

    try:
        raw, latency = client.generate(messages, temperature=LLM_TEMP_REASONING)
        parsed       = client.extract_json(raw)
        is_valid     = bool(parsed.get("valid", True))
        reason       = str(parsed.get("reason", ""))
        corrected    = parsed.get("corrected_query") or None
    except Exception as e:
        reason = f"validation fallback: {e}"

    step = TraceStep(
        step="integrity_validation",
        model=client.model_name,
        latency_ms=latency,
        result={"valid": is_valid, "reason": reason, "corrected_query": corrected},
        extra={"raw_output": raw[:200]},
    )
    return is_valid, reason, corrected, step
