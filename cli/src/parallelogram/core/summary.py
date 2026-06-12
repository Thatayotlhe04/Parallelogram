"""Dataset summary — the aggregate behind `parallelogram report`.

`check` answers "what is wrong, line by line". `report` answers "how healthy
is this dataset overall, and is it getting better or worse" — the question a
CI gate or a human deciding whether to train actually asks.

The summary is built from one validation pass plus a dry fix pass, so every
number here is the same number `check` and `check --fix` would produce; the
report can never disagree with the tools it summarizes.

Baseline comparison uses *rates*, not absolute counts, so a dataset that
grows from 1k to 10k records isn't punished for having more raw issues —
only for getting proportionally worse.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from .fixer import Fixer
from .report import Report, Severity
from .runner import Runner
from .rules import Rule

# Records whose token total lands in this top slice of the budget are
# "at risk": one chat-template change or tokenizer revision from overflow.
AT_RISK_FRACTION = 0.85

# Metrics compared against a baseline, as (key, human label) — all are
# rates in [0, 1] except clean_fraction, where *down* is worse.
_REGRESSION_RATES = [
    ("errors_per_record", "errors per record"),
    ("duplicates_per_record", "duplicate records per record"),
    ("over_budget_per_record", "records over token budget per record"),
    ("dropped_per_record", "records dropped by --fix per record"),
]


@dataclass
class DatasetSummary:
    """Everything `parallelogram report` knows about one dataset."""
    file: str = ""
    dataset_format: str = "openai-chat"
    disabled_rules: list[str] = field(default_factory=list)

    # record-level health
    total_records: int = 0
    clean_records: int = 0
    records_with_errors: int = 0
    records_with_warnings: int = 0
    unparseable_records: int = 0

    # issue-level counts
    error_count: int = 0
    warning_count: int = 0
    fixable_issue_count: int = 0
    issues_by_rule: dict[str, dict[str, int]] = field(default_factory=dict)

    # what `--fix --output` would do (computed via a dry fix pass)
    fix_projection: dict[str, int] = field(default_factory=dict)

    # token risk
    token_stats: dict[str, Any] = field(default_factory=dict)

    # duplicate clusters
    duplicate_stats: dict[str, int] = field(default_factory=dict)

    # shape of the data itself
    format_breakdown: dict[str, Any] = field(default_factory=dict)

    exit_code: int = 0

    # ── derived rates (used for baseline comparison) ───────────────────
    def rates(self) -> dict[str, float]:
        n = max(1, self.total_records)
        return {
            "clean_fraction": self.clean_records / n,
            "errors_per_record": self.error_count / n,
            "duplicates_per_record": self.duplicate_stats.get("duplicate_records", 0) / n,
            "over_budget_per_record": self.token_stats.get("over_budget", 0) / n,
            "dropped_per_record": self.fix_projection.get("dropped", 0) / n,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "format": self.dataset_format,
            "disabled_rules": self.disabled_rules,
            "summary": {
                "total_records": self.total_records,
                "clean_records": self.clean_records,
                "records_with_errors": self.records_with_errors,
                "records_with_warnings": self.records_with_warnings,
                "unparseable_records": self.unparseable_records,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "fixable_issue_count": self.fixable_issue_count,
                "exit_code": self.exit_code,
            },
            "issues_by_rule": self.issues_by_rule,
            "fix_projection": self.fix_projection,
            "token_stats": self.token_stats,
            "duplicate_stats": self.duplicate_stats,
            "format_breakdown": self.format_breakdown,
            "rates": self.rates(),
        }


def build_summary(
    path: str,
    rules: list[Rule],
    dataset_format: str = "openai-chat",
    disabled_rules: list[str] | None = None,
) -> DatasetSummary:
    """Run the full pipeline once and aggregate everything report shows."""
    runner = Runner(rules, dataset_format=dataset_format)
    report, parsed, _clean_raw, unparseable = runner.run_with_records(path)

    s = DatasetSummary(
        file=path,
        dataset_format=dataset_format,
        disabled_rules=sorted(disabled_rules or []),
        total_records=report.total_records,
        clean_records=report.valid_records,
        unparseable_records=len(unparseable),
        error_count=len(report.errors),
        warning_count=len(report.warnings),
        fixable_issue_count=sum(1 for i in report.issues if i.fixable),
    )

    # records (not issues) carrying each severity
    err_lines = {i.line_no for i in report.errors if i.line_no is not None}
    warn_lines = {i.line_no for i in report.warnings if i.line_no is not None}
    s.records_with_errors = len(err_lines)
    s.records_with_warnings = len(warn_lines)

    # per-rule breakdown
    by_rule: dict[str, dict[str, int]] = {}
    for i in report.issues:
        slot = by_rule.setdefault(i.rule_id, {"errors": 0, "warnings": 0, "fixable": 0})
        if i.severity == Severity.ERROR:
            slot["errors"] += 1
        elif i.severity == Severity.WARNING:
            slot["warnings"] += 1
        if i.fixable:
            slot["fixable"] += 1
    s.issues_by_rule = dict(sorted(by_rule.items()))

    # duplicate clusters — read BEFORE the fix pass (Fixer resets rule state)
    dup_rule = next((r for r in rules if r.id == "duplicates"), None)
    clusters = dup_rule.clusters if dup_rule is not None else []
    s.duplicate_stats = {
        "clusters": len(clusters),
        "duplicate_records": sum(len(c) - 1 for c in clusters),
        "largest_cluster": max((len(c) for c in clusters), default=0),
    }

    # token risk — same counter the context-window check used
    ctx_rule = next((r for r in rules if r.id == "context-window"), None)
    if ctx_rule is not None:
        exact, method = ctx_rule.counter_info()
        totals = [t for _, rec in parsed
                  if (t := ctx_rule.record_tokens(rec)) is not None]
        budget = ctx_rule.max_seq_len
        s.token_stats = {
            "counting": "exact" if exact else "estimated",
            "method": method,
            "max_seq_len": budget,
            "over_budget": sum(1 for t in totals if t > budget),
            "at_risk": sum(1 for t in totals
                           if AT_RISK_FRACTION * budget <= t <= budget),
            "median_tokens": int(statistics.median(totals)) if totals else 0,
            "max_tokens": max(totals, default=0),
        }

    # shape breakdown — computed on the normalized internal representation
    roles: dict[str, int] = {}
    turns: list[int] = []
    ends_on_assistant = 0
    for _, rec in parsed:
        if not isinstance(rec, dict):
            continue
        messages = rec.get("messages")
        if not isinstance(messages, list):
            continue
        turns.append(len(messages))
        last_role = None
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else None
            key = role if isinstance(role, str) else "invalid"
            roles[key] = roles.get(key, 0) + 1
            last_role = key
        if last_role == "assistant":
            ends_on_assistant += 1
    s.format_breakdown = {
        "declared_format": dataset_format,
        "roles": dict(sorted(roles.items())),
        "turns_per_record": {
            "min": min(turns, default=0),
            "median": int(statistics.median(turns)) if turns else 0,
            "max": max(turns, default=0),
        },
        "ends_on_assistant": ends_on_assistant,
    }

    # what --fix would do (dry pass; run LAST — it resets rule state)
    fixer = Fixer(rules)
    fr = fixer.fix(parsed, report.issues, unparseable)
    s.fix_projection = {
        "unchanged": fr.unchanged,
        "fixed": fr.fixed,
        "dropped": fr.dropped,
        "unparseable": fr.unparseable,
        "emitted": len(fr.clean_records),
    }

    # same exit semantics as `check` — the report never disagrees with it
    if report.has_errors:
        s.exit_code = 2
    elif report.has_warnings:
        s.exit_code = 1
    else:
        s.exit_code = 0
    return s


def compare_to_baseline(
    summary: DatasetSummary, baseline: dict[str, Any]
) -> list[str]:
    """Return human-readable regressions vs a previous `report --json`.

    Compares rates, so dataset growth alone never reads as regression.
    An empty list means no regression. Missing baseline fields are
    skipped rather than failed — older baselines stay usable.
    """
    regressions: list[str] = []
    base_rates = baseline.get("rates")
    if not isinstance(base_rates, dict):
        return ["baseline file has no 'rates' section — regenerate it with "
                "`parallelogram report <path> --json`"]

    now = summary.rates()
    eps = 1e-9

    base_clean = base_rates.get("clean_fraction")
    if isinstance(base_clean, (int, float)) and now["clean_fraction"] < base_clean - eps:
        regressions.append(
            f"clean fraction fell: {base_clean:.4f} → {now['clean_fraction']:.4f}"
        )

    for key, label in _REGRESSION_RATES:
        base_v = base_rates.get(key)
        if isinstance(base_v, (int, float)) and now[key] > base_v + eps:
            regressions.append(f"{label} rose: {base_v:.4f} → {now[key]:.4f}")

    return regressions
