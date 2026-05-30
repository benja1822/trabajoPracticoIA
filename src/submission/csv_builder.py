"""Build and validate the Kaggle submission CSV.

Format requirements (from spec):
- Columns: qid, preds
- qid: q1 … q40
- preds: exactly 10 image IDs separated by ";" (no trailing semicolon)
- IDs: no .jpg extension, no duplicates within a row
- Ordered best → worst (MAP@10 penalizes wrong order)
"""
import csv
from pathlib import Path
from typing import Dict, List, Optional

from src.config import SUBMISSION_FILE


class SubmissionBuilder:
    N_QUERIES = 40
    N_PREDS   = 10
    SEPARATOR  = ";"

    def __init__(self):
        self._rows: Dict[str, List[str]] = {}

    def add_query(self, qid: str, image_ids: List[str]) -> None:
        clean_ids = [iid.replace(".jpg", "").strip() for iid in image_ids]
        # Deduplicate preserving order
        seen, dedup = set(), []
        for iid in clean_ids:
            if iid not in seen:
                seen.add(iid)
                dedup.append(iid)
        self._rows[qid] = dedup[:self.N_PREDS]

    def validate(self) -> List[str]:
        errors = []
        expected_qids = {f"q{i}" for i in range(1, self.N_QUERIES + 1)}
        present_qids  = set(self._rows)

        missing = expected_qids - present_qids
        if missing:
            errors.append(f"Missing queries: {sorted(missing)}")

        for qid, ids in self._rows.items():
            if len(ids) != self.N_PREDS:
                errors.append(f"{qid}: expected {self.N_PREDS} preds, got {len(ids)}")
            if len(set(ids)) != len(ids):
                errors.append(f"{qid}: duplicate IDs detected")
            for iid in ids:
                if ".jpg" in iid:
                    errors.append(f"{qid}: ID {iid!r} still contains extension")

        return errors

    def save(self, path: Path = SUBMISSION_FILE) -> None:
        errors = self.validate()
        if errors:
            for e in errors:
                print(f"[WARNING] {e}")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["qid", "preds"])
            for i in range(1, self.N_QUERIES + 1):
                qid  = f"q{i}"
                ids  = self._rows.get(qid, [])
                preds = self.SEPARATOR.join(ids[:self.N_PREDS])
                writer.writerow([qid, preds])

        print(f"Submission saved → {path}  ({self.N_QUERIES} queries)")

    @classmethod
    def from_results(
        cls,
        results: Dict[str, List[str]],
        path: Optional[Path] = SUBMISSION_FILE,
    ) -> "SubmissionBuilder":
        builder = cls()
        for qid, ids in results.items():
            builder.add_query(qid, ids)
        if path:
            builder.save(path)
        return builder
