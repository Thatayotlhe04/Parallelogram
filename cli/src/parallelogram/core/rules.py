"""Rule base class and registry.

Rules declare their id, description, severity, and whether they are fixable.
Fixable rules implement either fix_record (per-record fixes — strip BOM,
truncate to context window, drop empty turns) or fix_dataset (cross-record
fixes — drop duplicate after first occurrence). Both default to no-ops, so
non-fixable rules need not implement them.
"""
from __future__ import annotations

from abc import ABC
from typing import Iterable, Any, Optional

from .report import Issue, Severity


class Rule(ABC):
    """Base class for validation rules.

    Subclasses set the class-level metadata and implement check_record.
    Rules with cross-record state (e.g., duplicates) override finalize.
    Rules that can auto-fix issues override fix_record or fix_dataset.
    """

    id: str = ""
    description: str = ""
    severity: Severity = Severity.ERROR
    fixable: bool = False  # whether issues from this rule can be auto-fixed

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        """Inspect a single parsed record. Yield any issues found."""
        return []

    def finalize(self) -> Iterable[Issue]:
        """Called once after all records have been processed.

        Used by rules that need to see the whole dataset before reporting,
        such as duplicate detection.
        """
        return []

    def reset(self) -> None:
        """Reset internal state. Called before each run."""
        return None

    # ── Fix interface (Phase 2 mechanical tier) ────────────────────────
    #
    # A fix can return either a corrected record (which then re-flows
    # through the runner for re-validation) or None (meaning "this record
    # is unfixable; drop it from the clean output"). Rules that cannot
    # fix their own issues leave both methods as no-ops, in which case
    # any record they flag stays an error and is dropped.

    def fix_record(self, record: Any, issue: Issue) -> Optional[Any]:
        """Per-record fix. Return the corrected record, or None to drop.

        The default returns the record unchanged, signalling "I have no
        opinion on this record." Combined with fixable=False on most
        rules, this means non-fixable rules don't accidentally pass
        broken records through.
        """
        return record

    def fix_dataset(self, records: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
        """Cross-record fix. Receives [(line_no, record), ...] and returns
        the same shape with whatever transformation makes sense. Default
        is identity. Used by duplicates (drop later occurrences) and
        could be extended to e.g. shuffle, balance, etc.
        """
        return records


class RuleRegistry:
    """Discovery surface for rule classes.

    Rules self-register with @registry.register so the CLI can enumerate them
    without each one being imported by name in cli.py.
    """

    def __init__(self) -> None:
        self._rules: dict[str, type[Rule]] = {}

    def register(self, rule_cls: type[Rule]) -> type[Rule]:
        if not rule_cls.id:
            raise ValueError(f"Rule {rule_cls.__name__} has no id")
        if rule_cls.id in self._rules:
            raise ValueError(f"Rule id {rule_cls.id!r} already registered")
        self._rules[rule_cls.id] = rule_cls
        return rule_cls

    def get(self, rule_id: str) -> type[Rule] | None:
        return self._rules.get(rule_id)

    def all(self) -> list[type[Rule]]:
        return list(self._rules.values())


registry = RuleRegistry()
