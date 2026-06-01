"""Unit tests for the Kaggle submission CSV builder.

Validates format requirements strictly:
- Exactly 40 queries (q1–q40)
- Exactly 10 IDs per query
- Semicolon separator, no trailing semicolon
- No .jpg extension, no duplicates within a row
"""
import csv
import pytest

from src.submission.csv_builder import SubmissionBuilder

pytestmark = pytest.mark.unit

FAKE_IDS = [f"2007_{i:06d}" for i in range(20)]


class TestSubmissionBuilder:

    def _full_builder(self) -> SubmissionBuilder:
        b = SubmissionBuilder()
        for i in range(1, 41):
            # Cycle through fake IDs to always have 10 unique ones
            ids = [FAKE_IDS[j % len(FAKE_IDS)] for j in range(i, i + 10)]
            # ensure uniqueness by using offset + distinct range
            ids = [f"2007_{(i * 100 + j):06d}" for j in range(10)]
            b.add_query(f"q{i}", ids)
        return b

    def test_valid_submission_has_no_errors(self):
        b = self._full_builder()
        assert b.validate() == []

    def test_add_query_strips_jpg_extension(self):
        b = SubmissionBuilder()
        b.add_query("q1", [f"2007_{i:06d}.jpg" for i in range(10)])
        ids = b._rows["q1"]
        assert all(".jpg" not in iid for iid in ids)

    def test_add_query_deduplicates(self):
        b = SubmissionBuilder()
        b.add_query("q1", ["2007_000001"] * 15)
        assert len(b._rows["q1"]) == 1

    def test_add_query_trims_to_10(self):
        b = SubmissionBuilder()
        b.add_query("q1", [f"2007_{i:06d}" for i in range(20)])
        assert len(b._rows["q1"]) == 10

    def test_validate_catches_missing_queries(self):
        b = SubmissionBuilder()
        b.add_query("q1", [f"2007_{i:06d}" for i in range(10)])
        errors = b.validate()
        assert any("Missing" in e for e in errors)

    def test_validate_catches_fewer_than_10_preds(self):
        b = self._full_builder()
        b._rows["q5"] = ["2007_000001", "2007_000002"]  # only 2
        errors = b.validate()
        assert any("q5" in e for e in errors)

    def test_validate_catches_jpg_extension(self):
        b = self._full_builder()
        b._rows["q1"] = [f"2007_{i:06d}.jpg" for i in range(10)]
        errors = b.validate()
        assert any("q1" in e for e in errors)

    def test_save_produces_correct_csv_format(self, tmp_path):
        b = self._full_builder()
        path = tmp_path / "submission.csv"
        b.save(path)

        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 40
        assert rows[0]["qid"] == "q1"

        # Check separator and count
        preds = rows[0]["preds"].split(";")
        assert len(preds) == 10

        # No trailing semicolon
        assert not rows[0]["preds"].endswith(";")

    def test_save_rows_in_order_q1_to_q40(self, tmp_path):
        b = self._full_builder()
        path = tmp_path / "submission.csv"
        b.save(path)
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        qids = [r["qid"] for r in rows]
        assert qids == [f"q{i}" for i in range(1, 41)]

    def test_from_results_classmethod(self, tmp_path):
        results = {f"q{i}": [f"2007_{(i*10+j):06d}" for j in range(10)] for i in range(1, 41)}
        path = tmp_path / "sub.csv"
        b = SubmissionBuilder.from_results(results, path)
        assert b.validate() == []
        assert path.exists()
