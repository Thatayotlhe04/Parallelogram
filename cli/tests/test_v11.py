"""1.1: Alpaca and preference formats, secrets, PII, and eval overlap."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.core.fixer import Fixer
from parallelogram.core.runner import Runner
from parallelogram.formats import get_parser, supported_formats
from parallelogram.formats.alpaca import normalize as alpaca_normalize
from parallelogram.formats.alpaca import source_metadata as alpaca_meta
from parallelogram.formats.alpaca import to_alpaca
from parallelogram.formats.preference import normalize as pref_normalize
from parallelogram.formats.preference import source_metadata as pref_meta
from parallelogram.formats.preference import to_preference
from parallelogram.rules import pii, secrets
from parallelogram.rules.context_window import ContextWindowRule
from parallelogram.rules.duplicates import record_hash
from parallelogram.rules.eval_overlap import EvalOverlapRule

runner = CliRunner()
U = lambda c: {"role": "user", "content": c}
A = lambda c: {"role": "assistant", "content": c}


def write(tmp_path, records, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return path


def check(path, *args):
    result = runner.invoke(app, ["check", str(path), "--json", *args])
    return result.exit_code, json.loads(result.stdout)


def rules_hit(payload):
    return sorted({i["rule"] for i in payload["issues"] if i["severity"] != "info"})


def fix_to(tmp_path, path, *args):
    out = tmp_path / "clean.jsonl"
    result = runner.invoke(app, ["check", str(path), "--fix", "--output", str(out), *args])
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    return result.exit_code, lines


# ── formats ───────────────────────────────────────────────────────────

def test_new_formats_are_supported():
    assert {"alpaca", "preference"} <= set(supported_formats())


def test_alpaca_normalizes_system_history_and_input():
    rec = {"instruction": "Sum", "input": "2 and 3", "output": "5", "system": "Be terse",
           "history": [["Hi", "Hello"]], "id": 7}
    assert alpaca_normalize(rec) == {"id": 7, "messages": [
        {"role": "system", "content": "Be terse"}, U("Hi"), A("Hello"), U("Sum\n\n2 and 3"), A("5")]}


@pytest.mark.parametrize("record,message", [
    ({"instruction": "Q"}, "Missing 'output' field"),
    ({"output": "A"}, "Missing 'instruction' field"),
    ({"instruction": "Q", "output": 3}, "'output' must be a string; found int"),
    ({"instruction": "Q", "output": "A", "history": [["x"]]}, "'history' must be a list of [user, assistant] string pairs"),
])
def test_alpaca_field_errors(tmp_path, record, message):
    code, payload = check(write(tmp_path, [record]))
    assert code == 2
    assert [i["message"] for i in payload["issues"] if i["rule"] == "schema"] == [message]


def test_alpaca_round_trip_restores_input_after_encoding_fix(tmp_path):
    path = write(tmp_path, [{"instruction": "Fix it", "input": "Donâ€™t stop", "output": "OK", "id": 1}])
    code, lines = fix_to(tmp_path, path)
    assert code == 0
    assert lines == [{"instruction": "Fix it", "input": "Don’t stop", "output": "OK", "id": 1}]


def test_alpaca_without_input_round_trips_exactly():
    rec = {"instruction": "Q", "output": "A"}
    assert to_alpaca(alpaca_normalize(rec), alpaca_meta(rec)) == rec


@pytest.mark.parametrize("rec", [
    {"prompt": "Q", "chosen": "Good", "rejected": "Bad"},
    {"prompt": [U("Q")], "chosen": [A("Good")], "rejected": [A("Bad")]},
    {"chosen": [U("Q"), A("Good")], "rejected": [U("Q"), A("Bad")]},
])
def test_preference_shapes_normalize_and_round_trip(rec):
    norm = pref_normalize(rec)
    assert norm == {"messages": [U("Q"), A("Good")], "rejected": [A("Bad")]}
    assert to_preference(norm, pref_meta(rec)) == rec


def test_preference_string_completions_need_a_prompt(tmp_path):
    code, payload = check(write(tmp_path, [{"chosen": "Good", "rejected": "Bad"}]))
    assert code == 2
    assert "Missing 'prompt' field" in payload["issues"][0]["message"]


def test_auto_detects_every_format(tmp_path):
    path = write(tmp_path, [
        {"messages": [U("a"), A("b")]},
        {"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "c"}]},
        {"prompt": "a", "completion": "d"},
        {"instruction": "a", "output": "e"},
        {"prompt": "a", "chosen": "f", "rejected": "g"},
    ])
    formats = [p.source_format for p in get_parser("auto")(str(path))]
    assert formats == ["openai-chat", "sharegpt", "prompt-completion", "alpaca", "preference"]
    assert check(path)[0] == 0


# ── preference-aware rules ────────────────────────────────────────────

def test_identical_chosen_and_rejected_is_an_error(tmp_path):
    code, payload = check(write(tmp_path, [{"prompt": "Q", "chosen": " Same ", "rejected": "Same"}]))
    assert code == 2 and rules_hit(payload) == ["preference"]


def test_rejected_side_is_checked_for_empty_content_and_roles(tmp_path):
    code, payload = check(write(tmp_path, [
        {"prompt": "Q", "chosen": "A", "rejected": "  "},
        {"prompt": [U("Q")], "chosen": [A("A")], "rejected": [U("oops")]},
    ]))
    msgs = [i["message"] for i in payload["issues"] if i["severity"] == "error"]
    assert code == 2
    assert any(m.startswith("Rejected response: Message 0") and "empty content" in m for m in msgs)
    assert any(m.startswith("Rejected response: Role alternation broken") for m in msgs)


def test_duplicates_tell_pairs_apart_by_rejected_answer():
    a = pref_normalize({"prompt": "Q", "chosen": "A", "rejected": "B"})
    b = pref_normalize({"prompt": "Q", "chosen": "A", "rejected": "C"})
    assert record_hash(a) != record_hash(b)


def test_context_window_measures_the_longer_side():
    rule = ContextWindowRule({"tokenizer": None, "max_seq_len": 50})
    rec = pref_normalize({"prompt": "Q", "chosen": "short", "rejected": "long " * 200})
    issues = [i for i in rule.check_record(rec, 1) if i.severity.value != "info"]
    assert issues and "max_seq_len" in issues[0].message


# ── secrets ───────────────────────────────────────────────────────────

KEYS = {
    "OpenAI API key": "sk-proj-Ab3dEf9hIjK2mNoPqR5sTuVwXyZ01234",
    "Anthropic API key": "sk-ant-api03-Zq8Lm2Xv7Rt4Wn9Ks1Pd6Hj3Fb",
    "AWS access key": "AKIAQ3ZJ7XK2M9PLW4TR",
    "GitHub token": "ghp_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5",
    "Hugging Face token": "hf_" + "Qw7Er9Ty2Ui4Op6As8Df1Gh3Jk5Lz7Xc",
    "Stripe secret key": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
}


@pytest.mark.parametrize("kind,key", KEYS.items())
def test_secrets_are_found_masked_and_redacted(kind, key):
    text = f"token: {key} end"
    (found_kind, m), = secrets.find_secrets(text)
    assert found_kind == kind
    assert key not in secrets.mask(m.group(0))
    assert secrets.redact(text) == "token: [REDACTED_SECRET] end"


def test_private_key_block_is_redacted_whole():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nIBAAKC\n-----END RSA PRIVATE KEY-----"
    assert secrets.redact(f"key:\n{block}\ndone") == "key:\n[REDACTED_SECRET]\ndone"


@pytest.mark.parametrize("text", [
    "use sk-YOUR_API_KEY_HERE", "AKIAIOSFODNN7EXAMPLE", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "pip install sk-learn-compatible-estimator-wrapper",
])
def test_placeholders_are_not_secrets(text):
    assert secrets.find_secrets(text) == []


def test_secrets_fail_the_check_and_never_print_the_key(tmp_path):
    key = KEYS["OpenAI API key"]
    result = runner.invoke(app, ["check", str(write(tmp_path, [{"messages": [U(f"my key {key}"), A("ok")]}]))])
    assert result.exit_code == 2
    assert key not in result.output


# ── PII ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,kind", [
    ("mail thato.k@gmail.com now", "email address"),
    ("call +267 77 123 456", "phone number"),
    ("call 212-867-5309", "phone number"),
    ("card 4539 1488 0343 6467", "payment card number"),
])
def test_pii_is_found(text, kind):
    assert [k for k, _ in pii.find_pii(text)] == [kind]


@pytest.mark.parametrize("text", [
    "write to support@example.com", "dev@api.test", "call 212-555-0199",
    "order 4539 1488 0343 6468", "version 1.2.3", "the year 2026",
])
def test_pii_false_positives_are_ignored(text):
    assert pii.find_pii(text) == []


def test_pii_is_a_warning_and_fix_redacts(tmp_path):
    path = write(tmp_path, [{"messages": [U("reach me at thato.k@gmail.com"), A("noted")]}])
    code, payload = check(path)
    assert code == 1 and rules_hit(payload) == ["pii"]
    code, lines = fix_to(tmp_path, path)
    assert code == 0
    assert lines[0]["messages"][0]["content"] == "reach me at [REDACTED_EMAIL]"


# ── eval overlap ──────────────────────────────────────────────────────

def test_eval_overlap_flags_full_and_prompt_matches(tmp_path):
    train = write(tmp_path, [
        {"messages": [U("same"), A("same")]},
        {"messages": [U("shared prompt"), A("train answer")]},
        {"messages": [U("unique"), A("fine")]},
    ])
    ev = write(tmp_path, [{"messages": [U("same"), A("same")]},
                          {"messages": [U("shared prompt"), A("eval answer")]}], name="eval.jsonl")
    code, payload = check(train, "--eval", str(ev))
    hits = {i["line"]: i["message"] for i in payload["issues"] if i["rule"] == "eval-overlap"}
    assert code == 2
    assert hits == {1: "Record also appears in the eval set (eval.jsonl:1)",
                    2: "Prompt also appears in the eval set (eval.jsonl:2)"}


def test_eval_overlap_is_off_without_eval(tmp_path):
    assert check(write(tmp_path, [{"messages": [U("q"), A("a")]}]))[0] == 0


def test_eval_overlap_fix_drops_train_records_and_credits_the_rule(tmp_path):
    train = write(tmp_path, [{"messages": [U("leak"), A("x")]}, {"messages": [U("ok"), A("y")]}])
    ev = write(tmp_path, [{"messages": [U("leak"), A("z")]}], name="eval.jsonl")
    rules = [EvalOverlapRule({"eval_path": str(ev)})]
    report, parsed, _, unparseable = Runner(rules).run_with_records(str(train))
    fr = Fixer(rules).fix(parsed, report.issues, unparseable)
    assert [ln for ln, _ in fr.clean_records] == [2]
    assert fr.fixes_by_rule == {"eval-overlap": 1}
    assert fr.outcomes[0].rules_fixed == ["eval-overlap"]


def test_report_accepts_eval(tmp_path):
    train = write(tmp_path, [{"messages": [U("leak"), A("x")]}])
    ev = write(tmp_path, [{"messages": [U("leak"), A("x")]}], name="eval.jsonl")
    result = runner.invoke(app, ["report", str(train), "--eval", str(ev), "--json"])
    assert result.exit_code == 2
    assert "eval-overlap" in json.loads(result.stdout)["issues_by_rule"]
