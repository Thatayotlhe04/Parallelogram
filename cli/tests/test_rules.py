"""Smoke tests for the rule set.

Not exhaustive — a unit-test pass for each rule and one end-to-end runner test.
Add per-rule tests as the rule set grows.
"""
from __future__ import annotations

import json

from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.core.report import Severity
from parallelogram.core.runner import Runner
from parallelogram.rules.schema import SchemaRule
from parallelogram.rules.roles import RolesRule
from parallelogram.rules.empty_content import EmptyContentRule
from parallelogram.rules.duplicates import DuplicatesRule
from parallelogram.rules.encoding import EncodingRule
from parallelogram.rules.context_window import ContextWindowRule
from parallelogram.core.tokenization import TokenCounter, resolve_counter
from parallelogram.formats.sharegpt import iter_jsonl as sharegpt_iter, to_sharegpt

try:
    # click < 8.2: stderr is mixed into stdout unless told otherwise
    cli_runner = CliRunner(mix_stderr=False)
except TypeError:
    # click >= 8.2: mix_stderr is gone; stderr is always captured separately
    cli_runner = CliRunner()


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


# Context window ---------------------------------------------------------------

def test_resolve_counter_falls_back_when_no_tokenizer():
    counter = resolve_counter(None)
    assert counter.exact is False
    assert counter.note
    # Length-based estimate: ~4 chars/token.
    assert counter.encode_len("a" * 40) == 10


def test_resolve_counter_claude_has_no_offline_tokenizer():
    counter = resolve_counter("claude-opus-4-8")
    assert counter.exact is False
    assert "count_tokens" in (counter.note or "")


def test_context_window_approximate_emits_warning_not_error():
    # No tokenizer → approximate counting → findings are warnings, never
    # errors (a heuristic shouldn't fail a CI gate or drop records). The
    # one-time advisory note is INFO — counting fidelity is advice, not a
    # data problem, and a clean dataset must exit 0 on a default install.
    rule = ContextWindowRule({"max_seq_len": 10})
    rule.reset()
    long_user = {"role": "user", "content": "word " * 200}
    issues = _check(rule, _record([long_user, {"role": "assistant", "content": "ok"}]))

    assert issues, "expected the over-long record to be flagged"
    assert not any(i.severity == Severity.ERROR for i in issues)
    note = [i for i in issues if i.line_no is None]
    assert note and all(i.severity == Severity.INFO for i in note)
    assert any("approximate" in i.message.lower() for i in note)
    findings = [i for i in issues if i.line_no is not None]
    assert findings and all(i.severity == Severity.WARNING for i in findings)
    assert any("estimated" in i.message.lower() for i in findings)


def test_context_window_approximate_note_emitted_once_per_run():
    rule = ContextWindowRule({"max_seq_len": 4096})
    rule.reset()
    rec = _record([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
    first = _check(rule, rec, line_no=1)
    second = _check(rule, rec, line_no=2)
    note_count = sum(1 for i in first + second if i.line_no is None)
    assert note_count == 1


def test_context_window_exact_emits_error():
    # Inject a fake exact tokenizer (1 token per whitespace-split word) so the
    # test needs no tiktoken/HF download. Exact counts surface as errors.
    rule = ContextWindowRule({"max_seq_len": 5})
    rule._counter = TokenCounter(
        encode_len=lambda t: len(t.split()), method="fake", exact=True,
    )
    rule._note_emitted = True  # exact path emits no note anyway

    issues = _check(rule, _record([
        {"role": "user", "content": "one two three four five six seven"},
        {"role": "assistant", "content": "ok"},
    ]))
    assert issues
    assert all(i.severity == Severity.ERROR for i in issues)
    assert "exceeds max_seq_len" in issues[0].message


def test_context_window_fix_truncates_to_fit():
    # Budget must leave room for the untouched assistant turn plus per-message
    # overhead, or the record is genuinely irrecoverable (and fix returns None).
    rule = ContextWindowRule({"max_seq_len": 14})
    rule._counter = TokenCounter(
        encode_len=lambda t: len(t.split()), method="fake", exact=True,
    )
    rule._note_emitted = True

    rec = _record([
        {"role": "user", "content": "one two three four five six seven eight"},
        {"role": "assistant", "content": "answer"},
    ])
    issue = _check(rule, rec)[0]
    fixed = rule.fix_record(rec, issue)
    assert fixed is not None
    assert rule._record_total(fixed["messages"]) <= rule.max_seq_len
    # The assistant turn (training target) is never touched.
    assert fixed["messages"][1]["content"] == "answer"


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


# ShareGPT format --------------------------------------------------------------


def _sharegpt_record(turns, system=None):
    rec = {"conversations": turns}
    if system is not None:
        rec["system"] = system
    return rec


def test_sharegpt_normalizes_roles(tmp_path):
    p = tmp_path / "sg.jsonl"
    p.write_text(
        json.dumps(_sharegpt_record(
            [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}],
            system="You are helpful.",
        )) + "\n",
        encoding="utf-8",
    )
    results = list(sharegpt_iter(str(p)))
    assert len(results) == 1
    msgs = results[0].record["messages"]
    # Top-level system is prepended; human→user, gpt→assistant.
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert [m["content"] for m in msgs] == ["You are helpful.", "Hi", "Hello"]


def test_sharegpt_unknown_from_passes_through_for_schema(tmp_path):
    # An unrecognized `from` must NOT be silently coerced — it should reach the
    # schema rule as an invalid role.
    p = tmp_path / "sg.jsonl"
    p.write_text(
        json.dumps(_sharegpt_record(
            [{"from": "wizard", "value": "Hi"}, {"from": "gpt", "value": "Hello"}],
        )) + "\n",
        encoding="utf-8",
    )
    runner = Runner([SchemaRule(), RolesRule()], dataset_format="sharegpt")
    report, clean = runner.run(str(p))
    assert report.has_errors
    assert not clean


def test_sharegpt_end_to_end_clean(tmp_path):
    p = tmp_path / "sg.jsonl"
    p.write_text(
        json.dumps(_sharegpt_record(
            [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}],
        )) + "\n"
        + json.dumps(_sharegpt_record(
            [{"from": "human", "value": "Capital of France?"}, {"from": "gpt", "value": "Paris."}],
        )) + "\n",
        encoding="utf-8",
    )
    runner = Runner(
        [SchemaRule(), RolesRule(), EmptyContentRule(), DuplicatesRule(), EncodingRule()],
        dataset_format="sharegpt",
    )
    report, clean = runner.run(str(p))
    assert report.total_records == 2
    assert report.valid_records == 2
    assert not report.has_errors


def test_sharegpt_is_accepted_by_default_auto_format(tmp_path):
    p = tmp_path / "qwen_sharegpt.jsonl"
    p.write_text(
        json.dumps(_sharegpt_record(
            [{"from": "user", "value": "Hi"}, {"from": "assistant", "value": "Hello"}],
        )) + "\n",
        encoding="utf-8",
    )
    result = cli_runner.invoke(app, ["check", str(p)])
    assert result.exit_code == 0, result.output


def test_auto_fix_round_trips_sharegpt_output(tmp_path):
    src = tmp_path / "qwen_sharegpt.jsonl"
    out = tmp_path / "fixed.jsonl"
    src.write_text(
        json.dumps(_sharegpt_record(
            [{"from": "human", "value": "Donâ€™t"}, {"from": "gpt", "value": "OK"}],
        )) + "\n",
        encoding="utf-8",
    )

    result = cli_runner.invoke(app, ["check", str(src), "--fix", "--output", str(out)])

    assert result.exit_code == 0, result.output
    fixed = json.loads(out.read_text(encoding="utf-8"))
    assert fixed == {
        "conversations": [
            {"from": "human", "value": "Don\u2019t"},
            {"from": "gpt", "value": "OK"},
        ]
    }


def test_to_sharegpt_round_trip():
    record = _record([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ])
    sg = to_sharegpt(record)
    assert sg == {"conversations": [
        {"from": "system", "value": "S"},
        {"from": "human", "value": "U"},
        {"from": "gpt", "value": "A"},
    ]}
