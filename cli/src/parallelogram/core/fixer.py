"""Fixer — orchestrates mechanical-tier fixes and re-validates.

The flow:
  1. Run a check pass to find issues (already done by Runner).
  2. For each record with fixable issues, ask the rules to fix it.
     Per-record fixes (empty-content, encoding, context-window) operate
     on individual records.
  3. Apply cross-record fixes (duplicates) to the surviving records.
  4. Re-run check on the fixed records to confirm cleanliness. Anything
     still erroring is by definition unfixable at the mechanical tier
     and is dropped from the clean output.
  5. Return a FixReport with per-record disposition and aggregate counts.

The fixer is deliberately conservative: when in doubt, drop the record
rather than emit something we're not sure about. Better to lose 5% of
records than to emit broken ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .report import Issue, Report, Severity
from .rules import Rule


class Disposition(str, Enum):
    UNCHANGED = "unchanged"   # was already clean
    FIXED = "fixed"           # had issues, fixed, now clean
    DROPPED = "dropped"       # had issues, couldn't fix, dropped
    UNPARSEABLE = "unparseable"  # JSON parse error — never fixable


@dataclass
class RecordOutcome:
    """Per-record disposition after fix attempt."""
    line_no: int
    disposition: Disposition
    rules_fixed: list[str] = field(default_factory=list)
    rules_unfixable: list[str] = field(default_factory=list)


@dataclass
class FixReport:
    total_records: int = 0
    unchanged: int = 0
    fixed: int = 0
    dropped: int = 0
    unparseable: int = 0
    outcomes: list[RecordOutcome] = field(default_factory=list)
    # Aggregate count of fixes by rule id, useful for usage analytics
    # and for surfacing what got fixed in the terminal output.
    fixes_by_rule: dict[str, int] = field(default_factory=dict)
    # Records that survived the fix pass (line_no, fixed_record).
    clean_records: list[tuple[int, Any]] = field(default_factory=list)


class Fixer:
    """Applies fix methods on rules and re-validates the output."""

    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self._rules_by_id = {r.id: r for r in rules}

    def fix(
        self,
        records: list[tuple[int, Any]],
        initial_issues: list[Issue],
        unparseable_lines: set[int],
    ) -> FixReport:
        """Run the fix pipeline.

        Args:
            records: parsed records as (line_no, record) — the runner's
                non-unparseable lines.
            initial_issues: issues found by the original check pass.
            unparseable_lines: lines that failed JSON parsing.
        """
        fr = FixReport()
        fr.total_records = len(records) + len(unparseable_lines)
        fr.unparseable = len(unparseable_lines)

        # Bucket issues by line_no so we can decide per record what to fix.
        issues_by_line: dict[int, list[Issue]] = {}
        for issue in initial_issues:
            if issue.line_no is None:
                continue
            issues_by_line.setdefault(issue.line_no, []).append(issue)

        # ── Stage 1: per-record fixes ───────────────────────────────────
        # Records with no issues sail through unchanged. Records with
        # only fixable issues get patched. Records with any unfixable
        # issue are dropped immediately — partial fixes aren't worth it.
        stage1: list[tuple[int, Any]] = []
        outcomes_partial: dict[int, RecordOutcome] = {}

        for line_no, record in records:
            issues = issues_by_line.get(line_no, [])
            if not issues:
                stage1.append((line_no, record))
                outcomes_partial[line_no] = RecordOutcome(
                    line_no=line_no,
                    disposition=Disposition.UNCHANGED,
                )
                continue

            # Decide if we can fix this record. Skip duplicates issues
            # here — they're handled in stage 2 (cross-record).
            non_dup_issues = [i for i in issues if i.rule_id != "duplicates"]
            unfixable_rules = [
                i.rule_id for i in non_dup_issues
                if not (self._rules_by_id.get(i.rule_id)
                        and self._rules_by_id[i.rule_id].fixable)
            ]

            if unfixable_rules:
                outcomes_partial[line_no] = RecordOutcome(
                    line_no=line_no,
                    disposition=Disposition.DROPPED,
                    rules_unfixable=unfixable_rules,
                )
                continue

            # All non-duplicate issues are fixable. Apply each rule's
            # fix in turn, threading the record through.
            current = record
            applied: list[str] = []
            failed = False
            for issue in non_dup_issues:
                rule = self._rules_by_id.get(issue.rule_id)
                if rule is None or not rule.fixable:
                    continue
                try:
                    new = rule.fix_record(current, issue)
                except Exception:  # noqa: BLE001 — defensive
                    failed = True
                    break
                if new is None:
                    failed = True
                    break
                if new is not current:
                    applied.append(rule.id)
                current = new

            if failed:
                outcomes_partial[line_no] = RecordOutcome(
                    line_no=line_no,
                    disposition=Disposition.DROPPED,
                    rules_unfixable=[i.rule_id for i in non_dup_issues],
                )
                continue

            stage1.append((line_no, current))
            for rid in applied:
                fr.fixes_by_rule[rid] = fr.fixes_by_rule.get(rid, 0) + 1
            outcomes_partial[line_no] = RecordOutcome(
                line_no=line_no,
                disposition=Disposition.FIXED if applied else Disposition.UNCHANGED,
                rules_fixed=applied,
            )

        # ── Stage 2: cross-record fixes ────────────────────────────────
        # Each rule with a fix_dataset method gets a chance to drop or
        # transform the surviving stage-1 records. duplicates is the
        # canonical example.
        stage2 = stage1
        kept_lines_before = {ln for ln, _ in stage2}
        for rule in self.rules:
            if not rule.fixable:
                continue
            try:
                stage2 = list(rule.fix_dataset(stage2))
            except Exception:  # noqa: BLE001 — defensive
                continue

        kept_lines_after = {ln for ln, _ in stage2}
        dropped_by_dataset = kept_lines_before - kept_lines_after
        for ln in dropped_by_dataset:
            # These were dropped by a cross-record rule — count as fixed
            # (the dataset is now correct) but record the line as dropped
            # so the user sees what happened.
            fr.fixes_by_rule["duplicates"] = fr.fixes_by_rule.get("duplicates", 0) + 1
            outcomes_partial[ln] = RecordOutcome(
                line_no=ln,
                disposition=Disposition.DROPPED,
                rules_fixed=["duplicates"],
            )

        # ── Stage 3: re-validate fixed records ─────────────────────────
        # Anything still erroring is unfixable at the mechanical tier.
        for rule in self.rules:
            rule.reset()

        final: list[tuple[int, Any]] = []
        rechecked_issues: dict[int, list[Issue]] = {}
        for line_no, record in stage2:
            for rule in self.rules:
                for issue in rule.check_record(record, line_no):
                    if issue.severity == Severity.ERROR:
                        rechecked_issues.setdefault(line_no, []).append(issue)
        for rule in self.rules:
            for issue in rule.finalize():
                if (issue.severity == Severity.ERROR
                        and issue.line_no is not None):
                    rechecked_issues.setdefault(issue.line_no, []).append(issue)

        for line_no, record in stage2:
            if line_no in rechecked_issues:
                # Survived stage 1 + 2 but still has errors → drop.
                still_bad = [i.rule_id for i in rechecked_issues[line_no]]
                prior = outcomes_partial.get(line_no)
                outcomes_partial[line_no] = RecordOutcome(
                    line_no=line_no,
                    disposition=Disposition.DROPPED,
                    rules_fixed=prior.rules_fixed if prior else [],
                    rules_unfixable=still_bad,
                )
                continue
            final.append((line_no, record))

        # ── Tally and return ────────────────────────────────────────────
        # Add unparseable lines to outcomes for reporting completeness.
        for ln in unparseable_lines:
            outcomes_partial[ln] = RecordOutcome(
                line_no=ln,
                disposition=Disposition.UNPARSEABLE,
            )

        # Sort outcomes by line number for stable, readable reports.
        fr.outcomes = sorted(outcomes_partial.values(), key=lambda o: o.line_no)
        for o in fr.outcomes:
            if o.disposition == Disposition.UNCHANGED:
                fr.unchanged += 1
            elif o.disposition == Disposition.FIXED:
                fr.fixed += 1
            elif o.disposition == Disposition.DROPPED:
                fr.dropped += 1
            # unparseable already tallied

        fr.clean_records = final
        return fr
