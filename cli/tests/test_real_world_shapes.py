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
from parallelogram.formats import auto

KEY = "sk-" + "AbCdEf0123456789" * 3


def write(tmp_path, *records, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def fix(path, tmp_path):
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    return result, out


def detected(path):
    return [r.source_format for r in auto.iter_jsonl(str(path))]


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


def multi_turn(prompt_turn):
    """A mlabonne/orpo-dpo-mix-40k row: ``prompt`` copies one user turn of a multi-turn pair."""
    turns = [{"role": "user", "content": "Describe Zephyria."},
             {"role": "assistant", "content": "A city in the clouds."},
             {"role": "user", "content": "Who lives there?"}]
    users = [t["content"] for t in turns if t["role"] == "user"]
    prompt = users[0] if prompt_turn == "first" else users[-1]
    return {"source": "Airoboros",
            "chosen": turns + [{"role": "assistant", "content": "Winged humanoids."}],
            "rejected": turns + [{"role": "assistant", "content": "Nobody."}],
            "prompt": prompt, "question": prompt}


@pytest.mark.parametrize("prompt_turn", ["first", "last"])
def test_multi_turn_pairs_whose_prompt_repeats_a_user_turn_are_clean(tmp_path, prompt_turn):
    path = write(tmp_path, multi_turn(prompt_turn))
    assert rules_hit(path) == set()
    result, out = fix(path, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == path.read_bytes()


def test_a_string_prompt_beside_full_conversations_is_metadata(tmp_path):
    # TRL's maybe_extract_prompt ignores it and splits the shared prefix; so do we.
    record = multi_turn("first")
    record["prompt"] = "<|user|>\nDescribe Zephyria.</s>\n<|assistant|>\n"
    path = write(tmp_path, record)
    assert rules_hit(path) == set()
    _, out = fix(path, tmp_path)
    assert out.read_bytes() == path.read_bytes()


def capybara_dpo(**overrides):
    """An argilla/distilabel-capybara-dpo-7k-binarized row: ``conversation`` is metadata."""
    record = multi_turn("first")
    record = {"source": "Airoboros",
              "conversation": [{"input": "Describe Zephyria.", "output": "A city in the clouds."}],
              "prompt": "<|user|>\nDescribe Zephyria.</s>",
              "chosen": record["chosen"], "rejected": record["rejected"],
              "rating_chosen": 5, "rating_rejected": 4}
    return {**record, **overrides}


def test_pairs_with_a_metadata_conversation_column_read_as_preference(tmp_path):
    path = write(tmp_path, capybara_dpo())
    assert detected(path) == ["preference"]
    assert rules_hit(path) == set()
    result, out = fix(path, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == path.read_bytes()
    leaked = write(tmp_path, capybara_dpo(rejected=capybara_dpo()["rejected"][:3] + [
        {"role": "assistant", "content": f"Use {KEY}"}]), name="leak.jsonl")
    assert "secrets" in rules_hit(leaked)


def test_conversation_turns_beside_chosen_still_read_as_sharegpt(tmp_path):
    record = {"conversations": [{"from": "human", "value": "2+2?"}],
              "chosen": [{"role": "assistant", "content": "4"}],
              "rejected": [{"role": "assistant", "content": "5"}]}
    assert detected(write(tmp_path, record)) == ["sharegpt"]


def test_issues_in_a_source_messages_column_name_it_messages(tmp_path):
    path = write(tmp_path, ultrafeedback(good=f"Use {KEY}"))
    messages = [i["message"] for i in check(path).to_dict()["issues"] if i["rule"] == "secrets"]
    assert any("Field 'messages[1].content'" in m for m in messages)
    assert not any("parallelogram:" in m for m in messages)
