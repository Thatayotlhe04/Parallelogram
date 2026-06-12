"""Tests for `parallelogram report` — summary aggregation, output formats,
exit codes, and the baseline regression gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.core.rules import registry
from parallelogram.core.summary import build_summary, compare_to_baseline

# Importing rule modules registers them (same trigger the CLI relies on).
from parallelogram.rules import (  # noqa: F401
    schema, roles, empty_content, context_window, duplicates, encoding,
)

try:
    # click < 8.2: stderr is mixed into stdout unless told otherwise
    runner = CliRunner(mix_stderr=False)
except TypeError:
    # click >= 8.2: mix_stderr is gone; stderr is always captured separately
    runner = CliRunner()


def _rules(tokenizer=None, max_seq_len=4096):
    out = []
    for rc in registry.all():
        if rc.id == "context-window":
            out.append(rc({"tokenizer": tokenizer, "max_seq_len": max_seq_len}))
        else:
            out.append(rc())
    return out


GOOD = '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}'
DUP = '{"messages": [{"role": "user", "content": "Same"}, {"role": "assistant", "content": "Twice"}]}'
ENDS_ON_USER = '{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "thanks"}]}'
MOJIBAKE = '{"messages": [{"role": "user", "content": "Don\\u00e2\\u20ac\\u2122t"}, {"role": "assistant", "content": "Fixed."}]}'


@pytest.fixture
def dirty(tmp_path: Path) -> Path:
    p = tmp_path / "dirty.jsonl"
    p.write_text("\n".join([GOOD, DUP, DUP, ENDS_ON_USER, MOJIBAKE, "not json"]) + "\n",
                 encoding="utf-8")
    return p


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    p = tmp_path / "clean.jsonl"
    p.write_text(GOOD + "\n", encoding="utf-8")
    return p


# ── aggregation ───────────────────────────────────────────────────────────

def test_summary_counts(dirty):
    s = build_summary(str(dirty), _rules())
    assert s.total_records == 6
    assert s.unparseable_records == 1
    # errors: duplicate (line 3), ends-on-user (line 4), invalid json (line 6)
    assert s.records_with_errors == 3
    assert s.clean_records == 3          # lines 1, 2, 5 (mojibake is a warning)
    assert s.records_with_warnings >= 1  # mojibake
    assert s.issues_by_rule["duplicates"]["errors"] == 1
    assert s.issues_by_rule["roles"]["errors"] == 1
    assert s.issues_by_rule["encoding"]["warnings"] == 1
    assert s.exit_code == 2


def test_summary_duplicate_clusters(dirty):
    s = build_summary(str(dirty), _rules())
    assert s.duplicate_stats == {
        "clusters": 1, "duplicate_records": 1, "largest_cluster": 2,
    }


def test_summary_fix_projection_matches_fixer_semantics(dirty):
    s = build_summary(str(dirty), _rules())
    fp = s.fix_projection
    # ends-on-user is unfixable → dropped; the duplicate line is dropped as
    # a fix; mojibake is repaired; line 6 never parses.
    assert fp["dropped"] == 2
    assert fp["fixed"] == 1
    assert fp["unparseable"] == 1
    assert fp["emitted"] == fp["unchanged"] + fp["fixed"]


def test_summary_token_stats_estimated(dirty):
    s = build_summary(str(dirty), _rules())
    assert s.token_stats["counting"] == "estimated"
    assert s.token_stats["max_seq_len"] == 4096
    assert s.token_stats["over_budget"] == 0
    assert s.token_stats["max_tokens"] > 0


def test_summary_format_breakdown(dirty):
    s = build_summary(str(dirty), _rules())
    fb = s.format_breakdown
    assert fb["detected_formats"] == {"openai-chat": 5}
    assert fb["roles"]["user"] >= 5
    assert fb["roles"]["assistant"] >= 4
    assert fb["turns_per_record"]["max"] == 3
    # GOOD, DUP, DUP, MOJIBAKE end on assistant; ENDS_ON_USER doesn't
    assert fb["ends_on_assistant"] == 4


def test_summary_detects_sharegpt_format(tmp_path):
    p = tmp_path / "sharegpt.jsonl"
    p.write_text(
        '{"conversations": [{"from": "human", "value": "Hi"}, '
        '{"from": "gpt", "value": "Hello"}]}\n',
        encoding="utf-8",
    )
    s = build_summary(str(p), _rules())
    assert s.format_breakdown["declared_format"] == "auto"
    assert s.format_breakdown["detected_formats"] == {"sharegpt": 1}
    assert s.format_breakdown["roles"] == {"assistant": 1, "user": 1}


# ── CLI surface ───────────────────────────────────────────────────────────

def test_report_exit_codes(dirty, clean):
    assert runner.invoke(app, ["report", str(clean)]).exit_code == 0
    assert runner.invoke(app, ["report", str(dirty)]).exit_code == 2


def test_report_json_is_machine_readable(dirty):
    r = runner.invoke(app, ["report", str(dirty), "--json"])
    payload = json.loads(r.stdout)
    assert payload["summary"]["total_records"] == 6
    assert payload["summary"]["exit_code"] == 2
    assert "rates" in payload and 0 <= payload["rates"]["clean_fraction"] <= 1
    assert r.exit_code == 2


def test_report_markdown(dirty):
    r = runner.invoke(app, ["report", str(dirty), "--markdown"])
    assert "## parallelogram report" in r.stdout
    assert "| rule | errors | warnings | fixable |" in r.stdout
    assert "`duplicates`" in r.stdout


def test_report_json_and_markdown_are_exclusive(dirty):
    r = runner.invoke(app, ["report", str(dirty), "--json", "--markdown"])
    assert r.exit_code == 2
    assert "mutually exclusive" in r.stderr


def test_report_out_writes_file(dirty, tmp_path):
    out = tmp_path / "report.json"
    r = runner.invoke(app, ["report", str(dirty), "--json", "--out", str(out)])
    assert out.exists()
    assert json.loads(out.read_text())["summary"]["total_records"] == 6
    assert r.exit_code == 2


# ── baseline regression gate ──────────────────────────────────────────────

def test_no_regression_against_own_baseline(dirty, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["report", str(dirty), "--json", "--out", str(base)])
    r = runner.invoke(app, ["report", str(dirty), "--baseline", str(base)])
    # same file vs itself: no regression — exit reflects dataset state (2)
    assert "no regression" in r.stderr
    assert r.exit_code == 2


def test_regression_exits_3(dirty, clean, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["report", str(clean), "--json", "--out", str(base)])
    r = runner.invoke(app, ["report", str(dirty), "--baseline", str(base)])
    assert r.exit_code == 3
    assert "regressed" in r.stderr


def test_improvement_is_not_regression(dirty, clean, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["report", str(dirty), "--json", "--out", str(base)])
    r = runner.invoke(app, ["report", str(clean), "--baseline", str(base)])
    assert r.exit_code == 0


def test_growth_alone_is_not_regression(tmp_path):
    small = tmp_path / "small.jsonl"
    small.write_text(GOOD + "\n", encoding="utf-8")
    big = tmp_path / "big.jsonl"
    variants = [
        GOOD.replace("Hi", f"Hi {i}").replace("Hello!", f"Hello {i}!")
        for i in range(50)
    ]
    big.write_text("\n".join(variants) + "\n", encoding="utf-8")
    base = tmp_path / "base.json"
    runner.invoke(app, ["report", str(small), "--json", "--out", str(base)])
    r = runner.invoke(app, ["report", str(big), "--baseline", str(base)])
    assert r.exit_code == 0


def test_bad_baseline_file_fails_loudly(dirty, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = runner.invoke(app, ["report", str(dirty), "--baseline", str(bad)])
    assert r.exit_code == 2
    assert "could not read baseline" in r.stderr


def test_compare_to_baseline_handles_old_format(dirty):
    s = build_summary(str(dirty), _rules())
    msgs = compare_to_baseline(s, {"summary": {}})  # no rates section
    assert msgs and "rates" in msgs[0]
