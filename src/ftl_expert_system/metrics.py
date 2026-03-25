"""Fast-path metrics for measuring self-improvement."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FastPathMetrics:
    """Track hit rate to measure whether the expert system is self-improving."""

    fast_path_hits: int = 0
    slow_path_falls: int = 0
    beliefs_extracted: int = 0

    @property
    def total_queries(self) -> int:
        return self.fast_path_hits + self.slow_path_falls

    @property
    def hit_rate(self) -> float:
        """Fraction of queries answered by the fast path."""
        if self.total_queries == 0:
            return 0.0
        return self.fast_path_hits / self.total_queries

    def record_fast_path(self) -> None:
        """Record a fast-path hit."""
        self.fast_path_hits += 1

    def record_slow_path(self, belief_extracted: bool = False) -> None:
        """Record a slow-path fallback."""
        self.slow_path_falls += 1
        if belief_extracted:
            self.beliefs_extracted += 1

    def save(self, path: Path) -> None:
        """Persist metrics to a JSON file."""
        path.write_text(json.dumps({
            "fast_path_hits": self.fast_path_hits,
            "slow_path_falls": self.slow_path_falls,
            "beliefs_extracted": self.beliefs_extracted,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> FastPathMetrics:
        """Load metrics from a JSON file."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            fast_path_hits=data.get("fast_path_hits", 0),
            slow_path_falls=data.get("slow_path_falls", 0),
            beliefs_extracted=data.get("beliefs_extracted", 0),
        )

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Queries: {self.total_queries} "
            f"(fast: {self.fast_path_hits}, slow: {self.slow_path_falls}) "
            f"| Hit rate: {self.hit_rate:.1%} "
            f"| Beliefs extracted: {self.beliefs_extracted}"
        )
