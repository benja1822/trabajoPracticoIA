"""Unit tests for evaluation metrics.

Covers MAP@10, AP@10, precision@k, recall@k, MRR.
All tests are pure logic — no I/O, no models.
"""
import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.unit


# ─── average_precision_at_k ──────────────────────────────────────────────────

class TestAveragePrecisionAtK:

    def test_all_relevant_returns_one(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = {"a", "b", "c", "d", "e"}
        assert average_precision_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_none_relevant_returns_zero(self):
        retrieved = ["a", "b", "c"]
        relevant  = {"x", "y", "z"}
        assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self):
        assert average_precision_at_k(["a", "b"], set(), k=5) == pytest.approx(0.0)

    def test_first_position_hit(self):
        # Only first result is relevant → AP = 1/1 / min(1,k) = 1.0
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = {"a"}
        assert average_precision_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_last_position_hit(self):
        # Only 5th result is relevant out of 5 → AP = (1/5) / min(1,5) = 0.2
        retrieved = ["b", "c", "d", "e", "a"]
        relevant  = {"a"}
        assert average_precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.2)

    def test_cutoff_respected(self):
        # Relevant item is at position 11 — outside k=10 cutoff
        retrieved = [str(i) for i in range(15)]
        relevant  = {"10"}
        assert average_precision_at_k(retrieved, relevant, k=10) == pytest.approx(0.0)

    @pytest.mark.parametrize("retrieved,relevant,k,expected", [
        # 2 relevant at positions 1,3 → (1/1 + 2/3) / 2
        (["a","b","c"],   {"a","c"},   3,  (1.0 + 2/3) / 2),
        # 1 relevant at position 2 → (1/2) / min(1,3)
        (["x","a","y"],   {"a"},       3,  0.5),
        # perfect top-2 of 3 relevant
        (["a","b","c","d","e"], {"a","b","c"}, 5, 1.0),
    ])
    def test_parametrized_cases(self, retrieved, relevant, k, expected):
        assert average_precision_at_k(retrieved, relevant, k) == pytest.approx(expected, rel=1e-4)


# ─── mean_average_precision ───────────────────────────────────────────────────

class TestMeanAveragePrecision:

    def test_single_perfect_query(self):
        results = {"q1": ["a", "b", "c"]}
        gt      = {"q1": {"a", "b", "c"}}
        assert mean_average_precision(results, gt, k=3) == pytest.approx(1.0)

    def test_missing_query_counts_as_zero(self):
        results = {}
        gt      = {"q1": {"a", "b"}}
        assert mean_average_precision(results, gt, k=10) == pytest.approx(0.0)

    def test_averages_over_all_queries(self):
        results = {
            "q1": ["a"],           # AP = 1.0 (relevant at pos 1)
            "q2": ["x", "x", "b"], # AP = 0.5 (relevant at pos 3, after dedup n/a — b is relevant)
        }
        gt = {"q1": {"a"}, "q2": {"b"}}
        # q1: AP = 1.0 / 1 = 1.0
        # q2: b at position 3, P@3 = 1/3, AP = (1/3) / 1 = 0.333...
        expected = (1.0 + 1/3) / 2
        assert mean_average_precision(results, gt, k=10) == pytest.approx(expected, rel=1e-3)

    def test_empty_ground_truth_returns_zero(self):
        assert mean_average_precision({}, {}, k=10) == pytest.approx(0.0)


# ─── precision_at_k ───────────────────────────────────────────────────────────

class TestPrecisionAtK:

    def test_all_relevant(self):
        assert precision_at_k(["a","b","c"], {"a","b","c"}, k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert precision_at_k(["a","b","c"], {"x","y","z"}, k=3) == pytest.approx(0.0)

    def test_half_relevant(self):
        assert precision_at_k(["a","x","b","y"], {"a","b"}, k=4) == pytest.approx(0.5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a","b"], {"a","b"}, k=0) == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self):
        # k=10 but only 3 results — precision uses k as denominator
        assert precision_at_k(["a","b","c"], {"a","b","c"}, k=10) == pytest.approx(0.3)


# ─── recall_at_k ──────────────────────────────────────────────────────────────

class TestRecallAtK:

    def test_perfect_recall(self):
        assert recall_at_k(["a","b","c"], {"a","b","c"}, k=3) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert recall_at_k(["x","y"], {"a","b"}, k=2) == pytest.approx(0.0)

    def test_partial_recall(self):
        # 2 out of 4 relevant retrieved in top-3
        assert recall_at_k(["a","x","b","c"], {"a","b","c","d"}, k=3) == pytest.approx(2/4)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a","b"], set(), k=5) == pytest.approx(0.0)


# ─── mean_reciprocal_rank ─────────────────────────────────────────────────────

class TestMeanReciprocalRank:

    def test_first_result_relevant(self):
        results = {"q1": ["a", "b", "c"]}
        gt      = {"q1": {"a"}}
        assert mean_reciprocal_rank(results, gt) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        results = {"q1": ["x", "a", "y"]}
        gt      = {"q1": {"a"}}
        assert mean_reciprocal_rank(results, gt) == pytest.approx(0.5)

    def test_no_result_relevant(self):
        results = {"q1": ["x", "y", "z"]}
        gt      = {"q1": {"a"}}
        assert mean_reciprocal_rank(results, gt) == pytest.approx(0.0)

    def test_averages_across_queries(self):
        results = {"q1": ["a"], "q2": ["x", "b"]}
        gt      = {"q1": {"a"}, "q2": {"b"}}
        assert mean_reciprocal_rank(results, gt) == pytest.approx((1.0 + 0.5) / 2)
