"""Compute accounting: one ledger for everything the experiment spends.

The unit is the forward-token-equivalent (FTE): one token through one forward
pass costs 1, and a backward pass over it costs `backward_multiplier` more
(2 by default, the usual rule of thumb). It is an accounting model, not a
FLOP measurement; wall-clock seconds are recorded next to it. What matters is
that every arm is charged by the same rule and that the optimizer's own
overhead (warmup, gradient scoring, selection, evaluation) is on the same
ledger as the training it is supposed to pay for.

Entries are keyed, so a resumed run that finds a finished stage does not
charge it twice, and a stage served from the cross-run cache is still charged
at its original cost with `cached: true`: reusing a result does not make it
free to have produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import read_json, write_json

STAGES = ("warmup", "scoring", "selection", "pilot_training", "evaluation", "final_evaluation")
# Stages that exist only because data is being optimized. Training and
# evaluating a subset would be paid by any approach; these would not.
OVERHEAD_STAGES = ("warmup", "scoring", "selection")


@dataclass(frozen=True)
class Cost:
    forward_tokens: int = 0
    backward_tokens: int = 0
    wall_seconds: float = 0.0

    def fte(self, backward_multiplier: float) -> float:
        return self.forward_tokens + backward_multiplier * self.backward_tokens

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.forward_tokens + other.forward_tokens, self.backward_tokens + other.backward_tokens,
                    self.wall_seconds + other.wall_seconds)


class Ledger:
    def __init__(self, path: Path, total_budget: float, backward_multiplier: float):
        self.path = Path(path)
        self.total_budget = total_budget
        self.backward_multiplier = backward_multiplier
        self.entries: dict[str, dict] = {}
        if self.path.is_file():
            self.entries = {e["key"]: e for e in read_json(self.path)["entries"]}

    def charge(self, key: str, stage: str, cost: Cost, *, trial: str | None = None,
               cached: bool = False) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown ledger stage {stage!r}")
        if key in self.entries:
            return
        self.entries[key] = {
            "key": key, "stage": stage, "trial": trial, "cached": cached,
            "forward_tokens": cost.forward_tokens, "backward_tokens": cost.backward_tokens,
            "fte": cost.fte(self.backward_multiplier), "wall_seconds": round(cost.wall_seconds, 3),
        }
        self.save()

    def save(self) -> None:
        write_json(self.path, {"unit": "forward-token-equivalents",
                               "backward_multiplier": self.backward_multiplier,
                               "entries": list(self.entries.values())})

    @property
    def total(self) -> float:
        return sum(e["fte"] for e in self.entries.values())

    @property
    def remaining(self) -> float:
        return self.total_budget - self.total

    def affords(self, estimate: float) -> bool:
        return self.total + estimate <= self.total_budget

    def sum(self, *, stage: str | None = None, trial: str | None = None, keys: tuple[str, ...] = ()) -> float:
        return sum(e["fte"] for e in self.entries.values()
                   if (stage is None or e["stage"] == stage) and (trial is None or e["trial"] == trial)
                   and (not keys or e["key"].startswith(keys)))

    def summary(self) -> dict:
        by_stage = {s: {"fte": 0.0, "wall_seconds": 0.0, "forward_tokens": 0, "backward_tokens": 0,
                        "from_cache_fte": 0.0} for s in STAGES}
        for e in self.entries.values():
            row = by_stage[e["stage"]]
            row["fte"] += e["fte"]
            row["wall_seconds"] += e["wall_seconds"]
            row["forward_tokens"] += e["forward_tokens"]
            row["backward_tokens"] += e["backward_tokens"]
            if e["cached"]:
                row["from_cache_fte"] += e["fte"]
        overhead = sum(by_stage[s]["fte"] for s in OVERHEAD_STAGES)
        for row in by_stage.values():
            row["wall_seconds"] = round(row["wall_seconds"], 3)
        return {
            "unit": "forward-token-equivalents (1 per forward token, "
                    f"+{self.backward_multiplier:g} per backward token); an accounting model, not measured FLOPs",
            "budget": self.total_budget, "total": self.total, "remaining": self.remaining,
            "by_stage": by_stage, "optimization_overhead": overhead,
            "optimization_overhead_share": (overhead / self.total) if self.total else 0.0,
            "wall_seconds": round(sum(e["wall_seconds"] for e in self.entries.values()), 3),
        }
