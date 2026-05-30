from typing import Dict, List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / k


def average_precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k     = retrieved[:k]
    score     = 0.0
    n_hits    = 0
    for i, img_id in enumerate(top_k, start=1):
        if img_id in relevant:
            n_hits += 1
            score  += n_hits / i
    return score / min(len(relevant), k)


def mean_average_precision(
    results: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 10,
) -> float:
    """Compute MAP@k over all queries present in ground_truth.

    Args:
        results:      {query_id: [img_id, ...]} ranked list of retrieved images.
        ground_truth: {query_id: {img_id, ...}} set of relevant images per query.
        k:            cutoff.
    """
    aps = []
    for qid, relevant in ground_truth.items():
        retrieved = results.get(qid, [])
        aps.append(average_precision_at_k(retrieved, relevant, k))
    return sum(aps) / len(aps) if aps else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(
    results: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    rrs = []
    for qid, relevant in ground_truth.items():
        retrieved = results.get(qid, [])
        rr = 0.0
        for rank, img_id in enumerate(retrieved, start=1):
            if img_id in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def per_query_report(
    results: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 10,
) -> List[Dict]:
    rows = []
    for qid in sorted(ground_truth):
        retrieved = results.get(qid, [])
        relevant  = ground_truth[qid]
        rows.append({
            "query_id": qid,
            "ap@10":    round(average_precision_at_k(retrieved, relevant, k), 4),
            "p@10":     round(precision_at_k(retrieved, relevant, k), 4),
            "recall@10": round(recall_at_k(retrieved, relevant, k), 4),
            "n_relevant": len(relevant),
        })
    return rows
