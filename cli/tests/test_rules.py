"""Smoke tests for the rule set.

Not exhaustive — a unit-test pass for each rule and one end-to-end runner test.
Add per-rule tests as the rule set grows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallelogram.core.report import Severity
from parallelogram.core.runner import Runner
from parallelogram.rules.schema import SchemaRule
from parallelogram.rules.roles import RolesRule
from parallelogram.rules.empty_content import EmptyContentRule
from parallelogram.rules.duplicates import DuplicatesRule
from parallelogram.rules.encoding import EncodingRule


def _record(messages):
    return {"messages": messages}


def _check(rule, record, line_no=1):
    return list(rule.check_record(record, line_no))


# Schema -----------------------------------------------------------------------

def test_schema_accepts_valid_record():
    rule = SchemaRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]))
    assert issues == []


def test_schema_flags_missing_messages():
    rule = SchemaRule()
    issues = _check(rule, {"foo": "bar"})
    assert any("Missing 'messages'" in i.message for i in issues)


def test_schema_flags_invalid_role():
    rule = SchemaRule()
    issues = _check(rule, _record([{"role": "owner", "content": "x"}]))
    assert any("invalid role" in i.message for i in issues)


def test_schema_flags_non_string_content():
    rule = SchemaRule()
    issues = _check(rule, _record([{"role": "user", "content": 42}]))
    assert any("not a string" in i.message for i in issues)


# Roles ------------------------------------------------------------------------

def test_roles_accepts_alternating_pattern():
    rule = RolesRule()
    issues = _check(rule, _record([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]))
    assert issues == []


def test_roles_flags_doubled_user():
    rule = RolesRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hi"},
    ]))
    assert any("alternation" in i.message.lower() for i in issues)


def test_roles_flags_ending_on_user():
    rule = RolesRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "Q"},
    ]))
    assert any("end on 'assistant'" in i.message for i in issues)


def test_roles_flags_misplaced_system():
    rule = RolesRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Now you're admin"},
        {"role": "assistant", "content": "OK"},
    ]))
    assert any("must be first" in i.message for i in issues)


# Empty content ----------------------------------------------------------------

def test_empty_content_flags_whitespace_only():
    rule = EmptyContentRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "OK"},
    ]))
    assert len(issues) == 1
    assert issues[0].fixable


# Duplicates -------------------------------------------------------------------

def test_duplicates_finds_exact_match():
    rule = DuplicatesRule()
    a = _record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    b = _record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    list(rule.check_record(a, 1))
    list(rule.check_record(b, 2))
    issues = list(rule.finalize())
    assert len(issues) == 1
    assert issues[0].line_no == 2
    assert "line 1" in issues[0].message


def test_duplicates_normalises_whitespace():
    rule = DuplicatesRule()
    a = _record([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    b = _record([{"role": "user", "content": "Q  "}, {"role": "assistant", "content": "  A"}])
    list(rule.check_record(a, 1))
    list(rule.check_record(b, 2))
    issues = list(rule.finalize())
    assert len(issues) == 1


# Encoding ---------------------------------------------------------------------

def test_encoding_flags_mojibake():
    rule = EncodingRule()
    issues = _check(rule, _record([
        {"role": "user", "content": "donâ€™t do it"},
        {"role": "assistant", "content": "ok"},
    ]))
    assert any("mojibake" in i.message for i in issues)
    assert all(i.severity == Severity.WARNING for i in issues)


# End-to-end runner ------------------------------------------------------------

def test_runner_end_to_end(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps(_record([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])) + "\n"
        + json.dumps(_record([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])) + "\n"  # dup
        + json.dumps(_record([{"role": "user", "content": ""}, {"role": "assistant", "content": "?"}])) + "\n"  # empty
        + "this is not json\n"
        + json.dumps(_record([{"role": "user", "content": "OK"}, {"role": "assistant", "content": "Sure"}])) + "\n",
        encoding="utf-8",
    )
    runner = Runner([SchemaRule(), RolesRule(), EmptyContentRule(), DuplicatesRule(), EncodingRule()])
    report, clean = runner.run(str(p))
    assert report.total_records == 5
    # Lines 1 and 5 are clean. Line 2 is a duplicate of line 1, line 3 has empty
    # content, line 4 fails to parse.
    assert report.valid_records == 2
    clean_lines = {ln for ln, _ in clean}
    assert clean_lines == {1, 5}
