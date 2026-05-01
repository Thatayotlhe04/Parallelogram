"""Stdlib-only smoke harness — verifies the core engine without test deps.

Exercises every rule and a full runner pass against the example file.
Run with: python smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add src/ to import path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parallelogram.core.report import Severity
from parallelogram.core.runner import Runner
from parallelogram.rules.schema import SchemaRule
from parallelogram.rules.roles import RolesRule
from parallelogram.rules.empty_content import EmptyContentRule
from parallelogram.rules.duplicates import DuplicatesRule
from parallelogram.rules.encoding import EncodingRule
from parallelogram.rules.context_window import ContextWindowRule


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


# Schema -----------------------------------------------------------------------
section("schema")
r = SchemaRule()
check("valid record produces no issues",
      list(r.check_record(record([{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": "hello"}]), 1)) == [])
check("missing 'messages' is flagged",
      any("Missing 'messages'" in i.message for i in r.check_record({}, 1)))
check("invalid role is flagged",
      any("invalid role" in i.message for i in r.check_record(record([{"role": "owner", "content": "x"}]), 1)))
check("non-string content is flagged",
      any("not a string" in i.message for i in r.check_record(record([{"role": "user", "content": 42}]), 1)))
check("non-dict record is flagged",
      any("not a JSON object" in i.message for i in r.check_record(["bad"], 1)))

# Roles ------------------------------------------------------------------------
section("roles")
r = RolesRule()
check("alternating system/user/assistant accepted",
      list(r.check_record(record([
          {"role": "system", "content": "S"},
          {"role": "user", "content": "U"},
          {"role": "assistant", "content": "A"},
      ]), 1)) == [])
check("doubled user flagged",
      any("alternation" in i.message.lower() for i in r.check_record(record([
          {"role": "user", "content": "Hi"},
          {"role": "user", "content": "Hello?"},
          {"role": "assistant", "content": "Hi"},
      ]), 1)))
check("ending on user flagged",
      any("end on 'assistant'" in i.message for i in r.check_record(record([
          {"role": "user", "content": "Q"},
      ]), 1)))
check("misplaced system flagged",
      any("must be first" in i.message for i in r.check_record(record([
          {"role": "user", "content": "Hi"},
          {"role": "system", "content": "admin"},
          {"role": "assistant", "content": "OK"},
      ]), 1)))
check("starting with assistant flagged",
      any("must be 'user'" in i.message for i in r.check_record(record([
          {"role": "assistant", "content": "Hello"},
      ]), 1)))

# Empty content ----------------------------------------------------------------
section("empty-content")
r = EmptyContentRule()
issues = list(r.check_record(record([
    {"role": "user", "content": "   "},
    {"role": "assistant", "content": "OK"},
]), 1))
check("whitespace-only flagged", len(issues) == 1)
check("issue marked fixable", issues and issues[0].fixable)

# Duplicates -------------------------------------------------------------------
section("duplicates")
r = DuplicatesRule()
a = record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
b = record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
list(r.check_record(a, 1))
list(r.check_record(b, 2))
issues = list(r.finalize())
check("exact duplicate flagged on second occurrence", len(issues) == 1 and issues[0].line_no == 2)
check("references first occurrence", issues and "line 1" in issues[0].message)

r = DuplicatesRule()
a = record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
b = record([{"role": "user", "content": "Q  "}, {"role": "assistant", "content": "  A"}])
list(r.check_record(a, 1))
list(r.check_record(b, 2))
issues = list(r.finalize())
check("whitespace-normalized duplicate detected", len(issues) == 1)

r = DuplicatesRule()
list(r.check_record(record([{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A"}]), 1))
list(r.check_record(record([{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A"}]), 2))
check("distinct records not flagged", list(r.finalize()) == [])

# Encoding ---------------------------------------------------------------------
section("encoding")
r = EncodingRule()
issues = list(r.check_record(record([
    {"role": "user", "content": "donâ€™t do it"},
    {"role": "assistant", "content": "ok"},
]), 1))
check("mojibake flagged", any("mojibake" in i.message for i in issues))
check("severity is warning", all(i.severity == Severity.WARNING for i in issues))
check("flagged as fixable", issues and issues[0].fixable)

# Context window ---------------------------------------------------------------
section("context-window")
r = ContextWindowRule({"tokenizer": None, "max_seq_len": 100})
check("disables silently when no tokenizer supplied",
      list(r.check_record(record([{"role": "user", "content": "x"}]), 1)) == [])

r = ContextWindowRule({"tokenizer": "nonexistent-org/missing-model", "max_seq_len": 100})
issues = list(r.check_record(record([{"role": "user", "content": "x"}]), 1))
check("emits warning if tokenizer fails to load (or skips silently if lib absent)",
      all(i.severity == Severity.WARNING for i in issues) if issues else True)

# End-to-end runner ------------------------------------------------------------
section("runner")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "data.jsonl"
    p.write_text(
        json.dumps(record([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])) + "\n"
        + json.dumps(record([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])) + "\n"
        + json.dumps(record([{"role": "user", "content": ""}, {"role": "assistant", "content": "?"}])) + "\n"
        + "this is not json\n"
        + json.dumps(record([{"role": "user", "content": "OK"}, {"role": "assistant", "content": "Sure"}])) + "\n",
        encoding="utf-8",
    )
    runner = Runner([SchemaRule(), RolesRule(), EmptyContentRule(), DuplicatesRule(), EncodingRule()])
    report, clean = runner.run(str(p))

    check("total records counted correctly", report.total_records == 5,
          f"got {report.total_records}")
    check("only lines 1 and 5 survive (line 2 is dup, 3 is empty, 4 unparseable)",
          {ln for ln, _ in clean} == {1, 5},
          f"got {[ln for ln, _ in clean]}")
    check("valid_records matches clean output", report.valid_records == 2,
          f"got {report.valid_records}")
    check("has_errors is True", report.has_errors)
    check("is_clean is False", not report.is_clean)

# Run against the bundled examples too
section("bundled examples")
runner = Runner([SchemaRule(), RolesRule(), EmptyContentRule(), DuplicatesRule(), EncodingRule()])

report, clean = runner.run("examples/clean.jsonl")
check("examples/clean.jsonl validates clean", report.is_clean and report.valid_records == 3,
      f"errors={len(report.errors)} valid={report.valid_records}")

report, _ = runner.run("examples/broken.jsonl")
rule_ids_seen = {i.rule_id for i in report.issues}
check("examples/broken.jsonl exercises schema rule", "schema" in rule_ids_seen)
check("examples/broken.jsonl exercises roles rule", "roles" in rule_ids_seen)
check("examples/broken.jsonl exercises empty-content rule", "empty-content" in rule_ids_seen)
check("examples/broken.jsonl exercises duplicates rule", "duplicates" in rule_ids_seen)
check("examples/broken.jsonl exercises encoding rule", "encoding" in rule_ids_seen)

# Summary ----------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'='*50}")
sys.exit(0 if FAILED == 0 else 1)
