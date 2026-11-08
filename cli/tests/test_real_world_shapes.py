"""Shapes popular public datasets actually have, found by running 1.6 over them.

Hugging Face ``datasets`` writes null for a column a row doesn't use, and
HuggingFaceH4/ultrafeedback_binarized keeps ``prompt`` and ``messages`` next to
full ``chosen``/``rejected`` conversations. Both must read as clean.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram import check
from parallelogram.cli import app

KEY = "sk-" + "AbCdEf0123456789" * 3


def write(tmp_path, *records, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def fix(path, tmp_path):
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    return result, out


def rules_hit(path):
    return {i["rule"] for i in check(path).to_dict()["issues"] if i["severity"] != "info"}


NULLABLE = [
    {"messages": [{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello", "tool_calls": None}]},
    {"messages": [{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello", "function_call": None}]},
    {"messages": [{"role": "system", "content": "s", "tool_calls": None},
                  {"role": "user", "content": "hi", "tool_call_id": None, "name": None},
                  {"role": "assistant", "content": "hello", "tool_calls": None}], "tools": None},
    {"messages": [{"role": "user", "content": "Lights on."},
                  {"role": "assistant", "content": None, "tool_calls": [
                      {"id": None, "type": "function", "function": {"name": "lights", "arguments": {"on": True}}}]},
                  {"role": "tool", "tool_call_id": None, "name": "lights", "content": "done"},
                  {"role": "assistant", "content": "Done.", "tool_calls": None}],
     "tools": [{"type": "function", "function": {"name": "lights", "parameters": {"type": "object"}}}]},
]


@pytest.mark.parametrize("record", NULLABLE)
def test_null_tool_fields_read_as_absent(tmp_path, record):
    path = write(tmp_path, record)
    assert rules_hit(path) == set()
    result, out = fix(path, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == path.read_bytes()


def test_empty_tool_calls_list_warns_and_is_kept_by_fix(tmp_path):
    record = {"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello", "tool_calls": []}]}
    path = write(tmp_path, record)
    issues = [i for i in check(path).to_dict()["issues"] if i["rule"] == "tool-calls"]
    assert [i["severity"] for i in issues] == ["warning"]
    assert check(path).exit_code == 1
    _, out = fix(path, tmp_path)
    assert out.read_bytes() == path.read_bytes()


def ultrafeedback(prompt="What is 2+2?", good="4", bad="5", system=False):
    head = [{"role": "system", "content": "Be brief."}] if system else []
    turn = [{"role": "user", "content": prompt}]
    chosen = head + turn + [{"role": "assistant", "content": good}]
    return {"prompt": prompt, "prompt_id": "abc", "chosen": chosen,
            "rejected": head + turn + [{"role": "assistant", "content": bad}],
            "messages": chosen, "score_chosen": 9.0, "score_rejected": 2.0}


@pytest.mark.parametrize("system", [False, True])
def test_ultrafeedback_binarized_rows_are_clean_and_round_trip(tmp_path, system):
    path = write(tmp_path, ultrafeedback(system=system))
    assert rules_hit(path) == set()
    result, out = fix(path, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == path.read_bytes()


def test_ultrafeedback_rows_still_get_checked(tmp_path):
    same = write(tmp_path, ultrafeedback(good="4", bad="4"), name="same.jsonl")
    assert "preference" in rules_hit(same)
    leaked = ultrafeedback(good=f"Use {KEY}")
    path = write(tmp_path, leaked, name="leak.jsonl")
    assert "secrets" in rules_hit(path)
    result, out = fix(path, tmp_path)
    fixed = json.loads(out.read_text())
    assert KEY not in out.read_text()
    assert list(fixed) == list(leaked)  # column order kept
    assert fixed["prompt"] == leaked["prompt"]


def test_a_prompt_that_is_not_repeated_is_still_prepended(tmp_path):
    record = {"prompt": "What is 2+2?",
              "chosen": [{"role": "assistant", "content": "4"}],
              "rejected": [{"role": "assistant", "content": "5"}]}
    path = write(tmp_path, record)
    assert rules_hit(path) == set()
    _, out = fix(path, tmp_path)
    assert json.loads(out.read_text()) == record
