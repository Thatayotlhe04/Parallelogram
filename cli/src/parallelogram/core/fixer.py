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

from .report import Issue, Severity, scrub
from .rules import Rule


# Order of record repairs, one kind per pass. Encoding first: stripping a BOM
# or repairing mojibake can join the halves of a key so the secrets pattern
# sees all of it. Then redaction, then anything that removes or cuts text,
# which could otherwise leave most of a key behind, too short to match.
# alpaca._split_user recognizes repaired text in this same order.
_FIX_ORDER = {"encoding": 0, "secrets": 1, "pii": 2, "empty-content": 3, "context-window": 4}

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
    # Unexpected exceptions raised by a rule's fix method. These used to be
    # swallowed silently — a crashing fixer would quietly drop a record (or
    # skip a whole dataset-level pass) with no trace. We now record them so
    # the CLI can surface them; a fixer bug should never hide behind a clean
    # exit code. Entries are human-readable "rule_id: message" strings.
    fixer_errors: list[str] = field(default_factory=list)


class Fixer:
    """Applies fix methods on rules and re-validates the output."""

    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self._rules_by_id = {r.id: r for r in rules}

    def _dataset_scoped(self, rule_id: str) -> bool:
        rule = self._rules_by_id.get(rule_id)
        return rule is not None and rule.fix_scope == "dataset"

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

            # Decide if we can fix this record. Skip dataset-scope issues
            # (duplicates, eval overlap) here — they're handled in stage 2.
            non_dup_issues = [i for i in issues if not self._dataset_scoped(i.rule_id)]
            # Only an unfixable *error* costs the record; a warning with no
            # mechanical repair (an empty tool_calls list) leaves it as it is.
            unfixable_rules = [
                i.rule_id for i in non_dup_issues
                if i.severity == Severity.ERROR
                and not (self._rules_by_id.get(i.rule_id)
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
            pending = non_dup_issues
            # Text repairs can expose another rule's match (e.g. a BOM inside
            # an email). Recheck until clean, with a bound for broken fixers.
            for _ in range(len(self.rules) + 1):
                if not pending:
                    break
                # One kind of repair per pass, then recheck. Redaction runs
                # before anything that cuts text: truncating first can leave
                # most of a key behind, too short to match any more.
                first = min(_FIX_ORDER.get(i.rule_id, len(_FIX_ORDER)) for i in pending)
                for issue in [i for i in pending
                              if _FIX_ORDER.get(i.rule_id, len(_FIX_ORDER)) == first]:
                    rule = self._rules_by_id.get(issue.rule_id)
                    if rule is None or not rule.fixable:
                        continue
                    try:
                        new = rule.fix_record(current, issue)
                    except Exception as exc:  # noqa: BLE001 — defensive
                        # Don't swallow: a crashing fix means we drop this record,
                        # but the user needs to know the fixer itself misbehaved.
                        fr.fixer_errors.append(scrub(
                            f"{rule.id}: fix_record raised on line {line_no}: "
                            f"{type(exc).__name__}: {exc}"
                        ))
                        failed = True
                        break
                    if new is None:
                        failed = True
                        break
                    if new is not current:
                        applied.append(rule.id)
                    current = new
                if failed:
                    break
                pending = [
                    issue for rule in self.rules
                    if rule.fixable and rule.fix_scope != "dataset"
                    for issue in rule.check_record(current, line_no)
                    if issue.severity != Severity.INFO
                ]
                if not pending:
                    break
            else:
                failed = True
                fr.fixer_errors.append(f"Record fixes did not converge on line {line_no}")

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
        dropped_by: dict[int, str] = {}
        for rule in self.rules:
            if not rule.fixable:
                continue
            before = {ln for ln, _ in stage2}
            try:
                stage2 = list(rule.fix_dataset(stage2))
                for ln in before - {ln for ln, _ in stage2}:
                    dropped_by[ln] = rule.id
            except Exception as exc:  # noqa: BLE001 — defensive
                # Previously this `continue` silently skipped the failed
                # dataset pass, leaving stage2 == the pre-pass records. The
                # re-validation in stage 3 still catches anything left broken,
                # but the failure itself must not be invisible.
                fr.fixer_errors.append(scrub(
                    f"{rule.id}: fix_dataset raised: "
                    f"{type(exc).__name__}: {exc}"
                ))
                continue

        for ln, rid in dropped_by.items():
            # Dropped by a cross-record rule — count as a fix (the dataset is
            # now correct) but record the line as dropped so the user sees
            # what happened, credited to the rule that dropped it.
            fr.fixes_by_rule[rid] = fr.fixes_by_rule.get(rid, 0) + 1
            outcomes_partial[ln] = RecordOutcome(
                line_no=ln,
                disposition=Disposition.DROPPED,
                rules_fixed=outcomes_partial[ln].rules_fixed + [rid],
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
