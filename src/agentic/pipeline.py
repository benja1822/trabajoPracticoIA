"""Main agentic orchestrator.

Wires together: language detection → translation → negation parsing →
semantic expansion → integrity validation → FAISS retrieval → reranking.

Each step is traced into a structured JSON log for auditability.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.agentic.tracer import AgentTrace, TraceStep
from src.agentic.translator import Translator
from src.agentic.negation_parser import parse_negations
from src.agentic.expander import expand_query
from src.agentic.validator import validate_query
from src.config import FAISS_TOP_K, FAISS_POOL_SIZE

if TYPE_CHECKING:
    from src.embeddings.clip_extractor import CLIPExtractor
    from src.indexing.faiss_manager import FAISSManager
    from src.reranking.negation_reranker import NegationReranker


@dataclass
class AgentResult:
    query_id: str
    original_query: str
    translated_query: str
    positive_query: str
    negative_attributes: List[str]
    expanded_query: str
    query_variants: List[str]
    is_valid: bool
    validation_reason: str
    corrected_query: Optional[str]
    final_query_used: str
    top_results: List[str]          # image IDs, ranked
    trace: AgentTrace

    @property
    def has_negation(self) -> bool:
        return bool(self.negative_attributes)


def _detect_language(text: str) -> tuple[str, TraceStep]:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(text)
    except Exception:
        lang = "es"
    step = TraceStep(step="language_detection", result=lang)
    return lang, step


class AgenticPipeline:
    def __init__(
        self,
        clip_extractor: "CLIPExtractor",
        faiss_manager: "FAISSManager",
        reranker: "NegationReranker",
        translator: Optional[Translator] = None,
        enable_llm: bool = True,
    ):
        self._clip       = clip_extractor
        self._faiss      = faiss_manager
        self._reranker   = reranker
        self._translator = translator or Translator()
        self._enable_llm = enable_llm

    @property
    def clip(self) -> "CLIPExtractor":
        return self._clip

    @property
    def faiss(self) -> "FAISSManager":
        return self._faiss

    def run(
        self,
        query: str,
        query_id: str = "q_unknown",
        top_k: int = FAISS_TOP_K,
        pool_size: int = FAISS_POOL_SIZE,
    ) -> AgentResult:
        trace = AgentTrace(query_id=query_id, original_query=query)

        # 1 — Detect language
        lang, lang_step = _detect_language(query)
        trace.add_step(lang_step)

        # 2 — Translate if not English
        if lang != "en":
            translated, trans_step = self._translator.translate(query)
            trace.add_step(trans_step)
        else:
            translated = query
            trace.add_step(TraceStep(step="translation", result="skipped (already English)"))

        # 3 — Negation detection
        neg_result, neg_step = parse_negations(translated)
        trace.add_step(neg_step)

        # 4 — Semantic expansion (LLM, optional)
        if self._enable_llm:
            expanded, variants, exp_step = expand_query(neg_result.positive_query)
            trace.add_step(exp_step)
        else:
            expanded = neg_result.positive_query
            variants = []
            trace.add_step(TraceStep(step="semantic_expansion", result="skipped (LLM disabled)"))

        # 5 — Integrity validation (LLM, optional)
        if self._enable_llm:
            is_valid, reason, corrected, val_step = validate_query(
                expanded, neg_result.negative_attributes
            )
            trace.add_step(val_step)
        else:
            is_valid, reason, corrected = True, "validation skipped", None
            trace.add_step(TraceStep(step="integrity_validation", result="skipped (LLM disabled)"))

        # Choose the best query to use for retrieval
        final_query = corrected if (corrected and not is_valid) else expanded

        # 6 — FAISS retrieval
        query_emb = self._clip.encode_text(final_query)
        candidates = self._faiss.search(query_emb, k=pool_size)
        trace.add_step(TraceStep(
            step="faiss_retrieval",
            result={"query_used": final_query, "pool_size": len(candidates)},
            extra={"top_5_ids": [c[0] for c in candidates[:5]]},
        ))

        # 7 — Reranking (only when there are negations)
        if neg_result.has_negation and neg_result.negative_attributes:
            reranked = self._reranker.rerank(
                candidates=candidates,
                negative_attributes=neg_result.negative_attributes,
            )
            trace.add_step(TraceStep(
                step="reranking",
                result={"applied": True, "negative_attributes": neg_result.negative_attributes},
            ))
        else:
            reranked = candidates
            trace.add_step(TraceStep(step="reranking", result={"applied": False}))

        top_results = [img_id for img_id, _ in reranked[:top_k]]
        trace.final_results = top_results
        trace.append_to_file()

        return AgentResult(
            query_id             = query_id,
            original_query       = query,
            translated_query     = translated,
            positive_query       = neg_result.positive_query,
            negative_attributes  = neg_result.negative_attributes,
            expanded_query       = expanded,
            query_variants       = variants,
            is_valid             = is_valid,
            validation_reason    = reason,
            corrected_query      = corrected,
            final_query_used     = final_query,
            top_results          = top_results,
            trace                = trace,
        )
