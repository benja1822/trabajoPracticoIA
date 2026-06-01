from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from src.config import TRACES_FILE


@dataclass
class TraceStep:
    step: str
    result: Any = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {"step": self.step}
        if self.result is not None:
            d["result"] = self.result
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 1)
        if self.model:
            d["model"] = self.model
        d.update(self.extra)
        return d


@dataclass
class AgentTrace:
    query_id: str
    original_query: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: List[TraceStep] = field(default_factory=list)
    final_results: List[str] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict:
        return {
            "query_id":       self.query_id,
            "timestamp":      self.timestamp,
            "original_query": self.original_query,
            "steps":          [s.to_dict() for s in self.steps],
            "final_results":  self.final_results,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def append_to_file(self, path: Path = TRACES_FILE) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_all(cls, path: Path = TRACES_FILE) -> List[Dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
