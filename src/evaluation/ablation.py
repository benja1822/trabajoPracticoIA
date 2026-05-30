"""Ablation study runner.

Compares three configurations:
  A — Baseline: FAISS search with original Spanish query (no processing)
  B — Reformulation: FAISS search with translated + LLM-expanded query
  C — Full pipeline: B + negation reranking

Outputs a comparison table and per-query breakdown.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from src.evaluation.metrics import (
    mean_average_precision, average_precision_at_k, per_query_report,
)

if TYPE_CHECKING:
    from src.embeddings.clip_extractor import CLIPExtractor
    from src.indexing.faiss_manager import FAISSManager
    from src.agentic.pipeline import AgenticPipeline
    from src.reranking.negation_reranker import NegationReranker


def run_baseline(
    queries: Dict[str, str],
    clip: "CLIPExtractor",
    faiss: "FAISSManager",
    k: int = 10,
) -> Dict[str, List[str]]:
    """Config A: encode Spanish query directly with CLIP, no reformulation."""
    results = {}
    for qid, query in queries.items():
        emb = clip.encode_text(query)
        hits = faiss.search(emb, k=k)
        results[qid] = [img_id for img_id, _ in hits]
    return results


def run_reformulation(
    queries: Dict[str, str],
    pipeline: "AgenticPipeline",
    k: int = 10,
) -> Dict[str, List[str]]:
    """Config B: LLM reformulation without reranking.

    Runs the full translation + expansion pipeline but skips the negation
    penalty step, isolating the contribution of query reformulation alone.
    """
    from src.agentic.translator import Translator
    from src.agentic.negation_parser import parse_negations
    from src.agentic.expander import expand_query

    translator = Translator()
    results = {}
    for qid, query in queries.items():
        translated, _ = translator.translate(query)
        neg_res, _    = parse_negations(translated)
        expanded, _, _ = expand_query(neg_res.positive_query)

        emb  = pipeline.clip.encode_text(expanded)
        hits = pipeline.faiss.search(emb, k=k)
        results[qid] = [img_id for img_id, _ in hits]
    return results


def run_full_pipeline(
    queries: Dict[str, str],
    pipeline: "AgenticPipeline",
    k: int = 10,
) -> Dict[str, List[str]]:
    """Config C: full pipeline including reranking."""
    results = {}
    for qid, query in queries.items():
        agent_result = pipeline.run(query, query_id=qid, top_k=k)
        results[qid] = agent_result.top_results
    return results


def compare_configurations(
    queries: Dict[str, str],
    ground_truth: Dict[str, Set[str]],
    results_a: Dict[str, List[str]],
    results_b: Dict[str, List[str]],
    results_c: Dict[str, List[str]],
    k: int = 10,
) -> Dict:
    map_a = mean_average_precision(results_a, ground_truth, k)
    map_b = mean_average_precision(results_b, ground_truth, k)
    map_c = mean_average_precision(results_c, ground_truth, k)

    per_query = []
    for qid in sorted(ground_truth):
        relevant = ground_truth[qid]
        per_query.append({
            "query_id":  qid,
            "query":     queries.get(qid, ""),
            "ap_a":      round(average_precision_at_k(results_a.get(qid, []), relevant, k), 4),
            "ap_b":      round(average_precision_at_k(results_b.get(qid, []), relevant, k), 4),
            "ap_c":      round(average_precision_at_k(results_c.get(qid, []), relevant, k), 4),
        })

    return {
        "map_a_baseline":      round(map_a, 4),
        "map_b_reformulation": round(map_b, 4),
        "map_c_full_pipeline": round(map_c, 4),
        "delta_a_to_b":        round(map_b - map_a, 4),
        "delta_b_to_c":        round(map_c - map_b, 4),
        "delta_a_to_c":        round(map_c - map_a, 4),
        "per_query":           per_query,
    }


def print_ablation_table(comparison: Dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {'Configuration':<30} {'MAP@10':>8}")
    print(f"{'─'*55}")
    print(f"  {'A — Baseline (direct CLIP)':<30} {comparison['map_a_baseline']:>8.4f}")
    print(f"  {'B — + LLM Reformulation':<30} {comparison['map_b_reformulation']:>8.4f}  Δ={comparison['delta_a_to_b']:+.4f}")
    print(f"  {'C — + Reranking':<30} {comparison['map_c_full_pipeline']:>8.4f}  Δ={comparison['delta_b_to_c']:+.4f}")
    print(f"{'─'*55}")
    print(f"  {'Total gain (A→C)':<30} {comparison['delta_a_to_c']:>+8.4f}")
    print(f"{'─'*55}\n")
