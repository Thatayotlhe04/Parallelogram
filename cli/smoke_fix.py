"""Stdlib-only smoke tests for the --fix mechanical tier.

Exercises every fixable rule's fix method, plus the Fixer orchestration
end-to-end against synthetic datasets. Run with: python smoke_fix.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parallelogram.core.fixer import Fixer, Disposition
from parallelogram.core.report import Issue, Severity
from parallelogram.core.runner import Runner
from parallelogram.rules.schema import SchemaRule
from parallelogram.rules.roles import RolesRule
from parallelogram.rules.empty_content import EmptyContentRule
from parallelogram.rules.duplicates import DuplicatesRule
from parallelogram.rules.encoding import EncodingRule


PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}  {detail}")


def section(title: str) -> None:
    print(f"\n[{title}]")


def record(messages):
    return {"messages": messages}


# ── encoding fix ────────────────────────────────────────────────────────
section("encoding rule — fix")

r = EncodingRule()
issue = Issue(rule_id="encoding", severity=Severity.WARNING, line_no=1,
              message="...", fixable=True)
fixed = r.fix_record(record([
    {"role": "user", "content": "donâ€™t do it"},
    {"role": "assistant", "content": "OK"},
]), issue)
check("mojibake â€™ replaced with U+2019",
      fixed["messages"][0]["content"] == "don\u2019t do it",
      f"got: {fixed['messages'][0]['content']!r}")

fixed = r.fix_record(record([
    {"role": "user", "content": "\ufeffhello"},
    {"role": "assistant", "content": "hi"},
]), issue)
check("BOM stripped", fixed["messages"][0]["content"] == "hello")

# Idempotence
fixed_twice = r.fix_record(fixed, issue)
check("encoding fix is idempotent", fixed == fixed_twice)

# Multiple mojibake patterns in one message
fixed = r.fix_record(record([
    {"role": "user", "content": "she said â€œhiâ€\x9d and donâ€™t leave"},
    {"role": "assistant", "content": "ok"},
]), issue)
check("multiple mojibake patterns all replaced",
      "â€" not in fixed["messages"][0]["content"],
      f"got: {fixed['messages'][0]['content']!r}")

# ── empty-content fix ──────────────────────────────────────────────────
section("empty-content rule — fix")

r = EmptyContentRule()
issue = Issue(rule_id="empty-content", severity=Severity.ERROR, line_no=1,
              message="...", fixable=True)

# Drop empty user turn, keep valid pair
fixed = r.fix_record(record([
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "   "},
]), issue)
check("empty turn dropped, valid pair retained",
      len(fixed["messages"]) == 2 and fixed["messages"][0]["role"] == "user",
      f"got: {fixed}")

# All empty → drop record
fixed = r.fix_record(record([
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "  "},
]), issue)
check("all-empty record returns None (drop)", fixed is None)

# ── duplicates fix ─────────────────────────────────────────────────────
section("duplicates rule — fix_dataset")

r = DuplicatesRule()
records = [
    (1, record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])),
    (2, record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])),
    (3, record([{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}])),
    (4, record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])),
]
out = r.fix_dataset(records)
check("first occurrence kept, dupes dropped",
      [ln for ln, _ in out] == [1, 3],
      f"got line numbers: {[ln for ln, _ in out]}")

# Whitespace-different dupes still detected
records = [
    (1, record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])),
    (2, record([{"role": "user", "content": "Q  "}, {"role": "assistant", "content": "  A"}])),
]
out = r.fix_dataset(records)
check("whitespace-normalized dupe also dropped", len(out) == 1)

# Distinct records preserved
records = [
    (1, record([{"role": "user", "content": "X"}, {"role": "assistant", "content": "A"}])),
    (2, record([{"role": "user", "content": "Y"}, {"role": "assistant", "content": "B"}])),
]
out = r.fix_dataset(records)
check("distinct records both preserved", len(out) == 2)

# ── Fixer end-to-end ───────────────────────────────────────────────────
section("Fixer orchestration")

# Setup: a dataset with one of each fixable type plus an unfixable record.
def good(line_q, line_a):
    return record([{"role": "user", "content": line_q},
                   {"role": "assistant", "content": line_a}])

dataset = [
    (1, good("Hi", "Hello")),                                              # clean
    (2, record([{"role": "user", "content": "donâ€™t leave"},              # encoding fixable
                {"role": "assistant", "content": "OK"}])),
    (3, record([{"role": "user", "content": "  "},                          # empty-content fixable
                {"role": "assistant", "content": "what?"}])),
    (4, good("Hi", "Hello")),                                              # duplicate of line 1
    (5, record([{"role": "user", "content": "Q"}])),                        # roles error - unfixable
    (6, good("Different", "Sure")),                                        # clean
]

# First, simulate running the runner manually to get issues for these records.
rules = [SchemaRule(), RolesRule(), EmptyContentRule(),
         DuplicatesRule(), EncodingRule()]
all_issues = []
for rule in rules:
    rule.reset()
for line_no, rec in dataset:
    for rule in rules:
        all_issues.extend(rule.check_record(rec, line_no))
for rule in rules:
    all_issues.extend(rule.finalize())

fixer = Fixer(rules)
fr = fixer.fix(dataset, all_issues, unparseable_lines=set())

check("fix report total matches input", fr.total_records == 6,
      f"got {fr.total_records}")

# Lines emitted: 1 (clean), 2 (encoding fix), 6 (clean).
# Line 3 should be dropped because dropping the empty user turn leaves
# the conversation starting with assistant (roles violation on re-check).
# Line 4 is a duplicate of line 1, dropped.
# Line 5 has unfixable roles error, dropped.
emitted_lines = sorted(ln for ln, _ in fr.clean_records)
check("lines 1, 2, 6 emitted; 3, 4, 5 dropped",
      emitted_lines == [1, 2, 6],
      f"got: {emitted_lines}")

# Encoding fix should have been applied to line 2's record
line2 = next(rec for ln, rec in fr.clean_records if ln == 2)
check("encoding fix applied (no mojibake in emitted record)",
      "â€" not in line2["messages"][0]["content"],
      f"got: {line2['messages'][0]['content']!r}")

# Outcome dispositions
outcomes_by_line = {o.line_no: o for o in fr.outcomes}
check("line 1 marked unchanged",
      outcomes_by_line[1].disposition == Disposition.UNCHANGED)
check("line 2 marked fixed",
      outcomes_by_line[2].disposition == Disposition.FIXED,
      f"got: {outcomes_by_line[2].disposition}")
check("line 4 marked dropped (duplicate)",
      outcomes_by_line[4].disposition == Disposition.DROPPED)
check("line 5 marked dropped (unfixable roles)",
      outcomes_by_line[5].disposition == Disposition.DROPPED)
check("line 6 marked unchanged",
      outcomes_by_line[6].disposition == Disposition.UNCHANGED)

# Fix counts
check("encoding fix counted", fr.fixes_by_rule.get("encoding", 0) == 1)
check("duplicates fix counted", fr.fixes_by_rule.get("duplicates", 0) >= 1)

# ── Atomicity (via the file-write helper) ──────────────────────────────
section("atomic write")

from parallelogram.core.io import atomic_write_jsonl

with tempfile.TemporaryDirectory() as td:
    target = Path(td) / "out.jsonl"
    pre = "ORIGINAL CONTENT\n"
    target.write_text(pre, encoding="utf-8")
    atomic_write_jsonl(target, ['{"a":1}', '{"b":2}'])
    content = target.read_text(encoding="utf-8")
    check("atomic write replaces original",
          content == '{"a":1}\n{"b":2}\n',
          f"got: {content!r}")
    # No leftover .tmp files
    leftover = [p for p in Path(td).iterdir() if p.suffix == ".tmp" or ".tmp" in p.name]
    check("no leftover .tmp files after success", not leftover,
          f"found: {leftover}")

# ── Re-validation catches partial fixes ────────────────────────────────
section("re-validation")

# Empty-content fix that leaves the conversation starting on assistant
# should be caught by re-validation and dropped.
rules_fresh = [SchemaRule(), RolesRule(), EmptyContentRule(),
               DuplicatesRule(), EncodingRule()]
for rule in rules_fresh:
    rule.reset()

ds = [
    (1, record([{"role": "user", "content": ""},  # empty user → drop turn → starts on assistant → roles violation
                {"role": "assistant", "content": "hello?"}])),
]
issues = []
for rule in rules_fresh:
    rule.reset()
for line_no, rec in ds:
    for rule in rules_fresh:
        issues.extend(rule.check_record(rec, line_no))
for rule in rules_fresh:
    issues.extend(rule.finalize())

fixer = Fixer(rules_fresh)
fr = fixer.fix(ds, issues, unparseable_lines=set())
check("partial fix caught by re-validation, record dropped",
      len(fr.clean_records) == 0 and fr.dropped == 1,
      f"clean={len(fr.clean_records)} dropped={fr.dropped}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'='*50}")
sys.exit(0 if FAILED == 0 else 1)
