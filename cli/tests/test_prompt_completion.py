"""Prompt/completion records share all validation and retain their output shape."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.formats import get_parser, supported_formats
from parallelogram.formats.prompt_completion import normalize, to_prompt_completion

runner = CliRunner()
USER = {"role": "user", "content": "Question"}
ASSISTANT = {"role": "assistant", "content": "Answer"}


def write_records(tmp_path, records):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("prompt", ["Question", [USER]])
@pytest.mark.parametrize("completion", ["Answer", [ASSISTANT]])
@pytest.mark.parametrize("format_id", ["auto", "prompt-completion"])
def test_normalization(tmp_path, prompt, completion, format_id):
    path = write_records(tmp_path, [{"prompt": prompt, "completion": completion, "extra": 42}])
    parsed, = get_parser(format_id)(str(path))
    assert parsed.record == {"messages": [USER, ASSISTANT]}
    assert parsed.source_format == "prompt-completion"
    assert "prompt-completion" in supported_formats()


def test_auto_mixed_formats_and_report(tmp_path):
    path = write_records(tmp_path, [
        {"prompt": "Q1", "completion": "A1"},
        {"messages": [USER, ASSISTANT]},
        {"conversations": [{"from": "human", "value": "Q3"}, {"from": "gpt", "value": "A3"}]},
    ])
    assert [r.source_format for r in get_parser("auto")(str(path))] == [
        "prompt-completion", "openai-chat", "sharegpt",
    ]
    result = runner.invoke(app, ["report", str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["format_breakdown"]["detected_formats"] == {
        "prompt-completion": 1, "openai-chat": 1, "sharegpt": 1,
    }


@pytest.mark.parametrize("key,value,expected", [
    ("messages", [USER, ASSISTANT], "openai-chat"),
    ("messages", None, "openai-chat"),
    ("conversations", [], "sharegpt"),
    ("conversation", [], "sharegpt"),
    ("conversations", None, "openai-chat"),
    ("conversation", None, "openai-chat"),
])
def test_existing_detection_precedence(tmp_path, key, value, expected):
    path = write_records(tmp_path, [{key: value, "prompt": "Q", "completion": "A"}])
    parsed, = get_parser("auto")(str(path))
    assert parsed.source_format == expected


@pytest.mark.parametrize("format_id", ["auto", "prompt-completion"])
@pytest.mark.parametrize("record,message", [
    ({"prompt": "Q"}, "Missing 'completion' field"),
    ({"completion": "A"}, "Missing 'prompt' field"),
    ({"prompt": "Q", "completion": None}, "'completion' must be a string or list"),
    ({"prompt": None, "completion": "A"}, "'prompt' must be a string or list"),
    ({"prompt": "Q", "completion": 1}, "'completion' must be a string or list"),
    ({"prompt": {}, "completion": "A"}, "'prompt' must be a string or list"),
    ({"prompt": False, "completion": "A"}, "'prompt' must be a string or list"),
    ({"prompt": [42], "completion": "A"}, "Message 0 is not an object"),
    ({"prompt": "Q", "completion": [{"role": "assistant"}]}, "missing 'content' field"),
])
def test_malformed_fields(tmp_path, format_id, record, message):
    path = write_records(tmp_path, [record])
    for command in ("check", "report"):
        result = runner.invoke(app, [command, str(path), "--format", format_id, "--json"])
        assert result.exit_code == 2, result.output
        if command == "check":
            issues = json.loads(result.stdout)["issues"]
            assert any(i["rule"] == "schema" and message in i["message"] for i in issues)
            assert not any("Missing 'messages'" in i["message"] for i in issues)


@pytest.mark.parametrize("record,rule,code,options", [
    ({"prompt": "Q", "completion": " "}, "empty-content", 2, []),
    ({"prompt": "Q" * 1000, "completion": "A"}, "context-window", 1, ["--max-seq-len", "50"]),
    ({"prompt": [USER, ASSISTANT], "completion": [USER]}, "roles", 2, []),
    ({"prompt": "\ufeffQ", "completion": "A"}, "encoding", 1, []),
])
def test_rules_fire(tmp_path, record, rule, code, options):
    path = write_records(tmp_path, [record])
    result = runner.invoke(app, ["check", str(path), "--json", *options])
    assert result.exit_code == code, result.output
    assert any(i["rule"] == rule and i["line"] == 1 for i in json.loads(result.stdout)["issues"])
    report = runner.invoke(app, ["report", str(path), "--json", *options])
    assert report.exit_code == code, report.output


def test_duplicate_pairs(tmp_path):
    path = write_records(tmp_path, [
        {"prompt": "Question", "completion": "Answer"},
        {"prompt": [USER], "completion": [ASSISTANT]},
    ])
    result = runner.invoke(app, ["check", str(path), "--json"])
    assert result.exit_code == 2
    assert any(i["rule"] == "duplicates" and i["line"] == 2 for i in json.loads(result.stdout)["issues"])


@pytest.mark.parametrize("string_prompt", [False, True])
@pytest.mark.parametrize("string_completion", [False, True])
@pytest.mark.parametrize("format_id", ["auto", "prompt-completion"])
def test_fix_round_trip(tmp_path, string_prompt, string_completion, format_id):
    prompt = "\ufeffQuestion" if string_prompt else [{**USER, "content": "\ufeffQuestion"}]
    completion = "\ufeffAnswer" if string_completion else [{**ASSISTANT, "content": "\ufeffAnswer"}]
    path = write_records(tmp_path, [{"prompt": prompt, "completion": completion, "extra": {"ignored": True}}])
    out = tmp_path / "fixed.jsonl"
    result = runner.invoke(app, ["check", str(path), "--format", format_id, "--fix", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text()) == {
        "prompt": "Question" if string_prompt else [USER],
        "completion": "Answer" if string_completion else [ASSISTANT],
    }
    assert runner.invoke(app, ["check", str(out)]).exit_code == 0


def test_conversational_split_uses_trailing_assistants():
    messages = [USER, ASSISTANT, {**USER, "content": "Again"}, ASSISTANT, ASSISTANT]
    assert to_prompt_completion({"messages": messages}) == {
        "prompt": messages[:3], "completion": messages[3:],
    }


def test_fix_conversational_multiturn(tmp_path):
    path = write_records(tmp_path, [{
        "prompt": [{"role": "system", "content": "Help."}, USER],
        "completion": [ASSISTANT, {**USER, "content": "\ufeffAgain"}, ASSISTANT],
    }])
    out = tmp_path / "fixed.jsonl"
    result = runner.invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    assert result.exit_code == 0, result.output
    fixed = json.loads(out.read_text())
    assert [m["role"] for m in fixed["prompt"]] == ["system", "user", "assistant", "user"]
    assert fixed["completion"] == [ASSISTANT]
    assert runner.invoke(app, ["check", str(out)]).exit_code == 0


def test_blank_lines_and_invalid_json(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('\n{"prompt": "Q", "completion": "A"}\nnot json\n')
    parsed = list(get_parser("prompt-completion")(str(path)))
    assert [p.line_no for p in parsed] == [2, 3]
    assert parsed[1].parse_error
    assert parsed[1].source_format == "prompt-completion"


@pytest.mark.parametrize("record", [None, [], 1, {"prompt": "Q"}])
def test_normalizer_and_inverse_preserve_invalid_records(record):
    assert normalize(record) == record
    assert to_prompt_completion(record) == record


def test_explicit_chat_does_not_silently_accept_unnormalized_pairs(tmp_path):
    path = write_records(tmp_path, [{"prompt": "Question", "completion": "Answer"}])
    result = runner.invoke(app, ["check", str(path), "--format", "openai-chat"])
    assert result.exit_code == 2
