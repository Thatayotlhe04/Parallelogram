"""Regression tests for the independent 1.5.0 leak audit.

Key-shaped strings are assembled at runtime so the repository itself holds
nothing a secret scanner should flag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from parallelogram import check, fix
from parallelogram.cli import app
from parallelogram.mcp_server import serve

GH = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
EMAIL = "pg-audit-7f3a@parallelogram.dev"
runner = CliRunner()


def chat(*turns, **extra):
    roles = ["user", "assistant"]
    return {**extra, "messages": [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]}


def write(path: Path, *records) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records))
    return path


# ── 1. every string field is scanned and redacted ─────────────────────────────

@pytest.mark.parametrize("record", [
    chat("hi", "hello", id=GH),
    chat("hi", "hello", metadata={"source": {"token": GH}}),
    {"messages": [{"role": "user", "content": "hi", "name": GH}, {"role": "assistant", "content": "ok"}]},
    {"messages": [{"role": "user", "content": "call it"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": json.dumps({"k": GH})}}]},
                  {"role": "tool", "content": "done"}, {"role": "assistant", "content": "ok"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "key " + GH}]},
                  {"role": "assistant", "content": "ok"}]},
    {"instruction": "Say hi", "output": "hi", "id": GH},
    {"prompt": "Q", "chosen": "A", "rejected": "B", "source": GH},
])
def test_secret_outside_content_is_found_and_redacted(tmp_path, record):
    src = write(tmp_path / "in.jsonl", record)
    result = check(src)
    assert result.exit_code == 2
    assert any(i["rule"] == "secrets" for i in result.to_dict()["issues"])
    assert GH not in json.dumps(result.to_dict())
    out = tmp_path / "out.jsonl"
    fix(src, output=out)
    assert GH not in out.read_text()


def test_pii_is_checked_in_training_text_not_ids_or_metadata(tmp_path):
    """Second audit pass: PII in every field flagged SKUs, UUIDs and git
    remotes and rewrote them. PII is checked where a model learns it."""
    records = [
        {"instruction": "Write to " + EMAIL, "output": "ok", "contact": EMAIL},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "call +1 415 867 5309"}]},
                      {"role": "assistant", "content": "ok"}]},
    ]
    src = write(tmp_path / "in.jsonl", *records)
    issues = [i for i in check(src).to_dict()["issues"] if i["rule"] == "pii"]
    assert {i["line"] for i in issues} == {1, 2}
    out = tmp_path / "out.jsonl"
    fix(src, output=out)
    first = json.loads(out.read_text().splitlines()[0])
    assert first["instruction"] == "Write to [REDACTED_EMAIL]" and first["contact"] == EMAIL
    assert "415 867 5309" not in out.read_text()


def test_ordinary_structured_fields_stay_clean(tmp_path):
    import uuid
    rows = []
    for i in range(200):
        args = json.dumps({"order": f"220-611-{5000 + i:04d}", "ref": 4539148803436467 + i})
        rows.append({
            "id": str(uuid.UUID(int=i)), "num_id": f"{4000000000000000 + i * 7919:016d}",
            "repo": "git@github.com:org/x.git", "sku": f"220-611-{5000 + i:04d}",
            "metadata": {"tool": "lookup", "arguments": args},
            "messages": [{"role": "user", "content": "Where is my order?"},
                         {"role": "assistant", "content": f"It shipped. ({i})"}]})
    src = write(tmp_path / "in.jsonl", *rows)
    assert check(src).exit_code == 0
    out = tmp_path / "out.jsonl"
    fix(src, output=out)
    assert [json.loads(l) for l in out.read_text().splitlines()] == rows


@pytest.mark.parametrize("record", [
    {"prompt": "p", "completion": "c", "id": GH},
    {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}], "id": GH},
    chat("hi", "hello", metadata={"ratings": {GH: 1}}),
])
def test_secrets_in_sharegpt_pc_extras_and_dict_keys(tmp_path, record):
    src = write(tmp_path / "in.jsonl", record)
    assert check(src).exit_code == 2
    out = tmp_path / "out.jsonl"
    fix(src, output=out)
    text = out.read_text()
    assert GH not in text and "[REDACTED_SECRET]" in text
    kept = json.loads(text)
    assert set(kept) == set(record)  # extras written back, nothing dropped


def test_key_split_by_a_bom_is_redacted_whole(tmp_path):
    """Second audit pass: redaction ran before the encoding repair, so a BOM
    inside a key left its tail behind."""
    key = "sk-ant-api03-" + "Ab3Cd5Ef7Gh9Ij2Kl4Mn6Op8" + "\ufeff" + "Qr1St2Uv3Wx4Yz5Ab6Cd7Ef8Gh9Ij0Kl1Mn2Op3Qr4St5Uv6Wx7Yz8Ab9Cd0Ef1Gh2Ij3Kl4Mn5" + "AA"
    for record in (chat("my key is " + key, "ok"),
                   {"instruction": "use " + key, "input": "keep me separate", "output": "ok"}):
        src = write(tmp_path / "in.jsonl", record)
        out = tmp_path / "out.jsonl"
        fix(src, output=out)
        text = out.read_text()
        assert "Qr1St2Uv3Wx4" not in text and "[REDACTED_SECRET]" in text, text
        if "instruction" in record:
            assert json.loads(text)["input"] == "keep me separate"
        assert check(out).exit_code == 0


def test_email_mask_hides_the_domain(tmp_path):
    src = write(tmp_path / "in.jsonl", chat("Mail " + EMAIL, "ok"))
    messages = [i["message"] for i in check(src).to_dict()["issues"] if i["rule"] == "pii"]
    assert messages and all("parallelogram.dev" not in m for m in messages)
    assert any("p***@***.dev" in m for m in messages)


# ── 2. hostile lines are invalid JSON, never a crash ──────────────────────────

@pytest.mark.parametrize("line", [
    '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}],"n":' + "9" * 4301 + "}",
    "[" * 100_000 + "]" * 100_000,
])
@pytest.mark.parametrize("command", [["check"], ["check", "--json"], ["report"], ["report", "--markdown"],
                                     ["check", "--fix", "--dry-run"]])
def test_hostile_line_is_reported_not_crashed(tmp_path, line, command):
    src = write(tmp_path / "in.jsonl", chat("a", "b"), line)
    result = runner.invoke(app, [*command, str(src)])
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code in (1, 2), result.output  # 2 for check/report, 1 for a partial fix


def test_unexpected_crash_exits_3_and_is_scrubbed():
    code = ("import parallelogram.cli as c\n"
            "def boom(): raise RuntimeError('boom ' + 'ghp_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8')\n"
            "c.app = boom\nc.run()\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 3
    assert "internal error" in done.stderr and "Traceback" not in done.stderr
    assert GH not in done.stderr


# ── 3. redaction happens before truncation ────────────────────────────────────

def test_secret_is_redacted_before_truncation(tmp_path):
    src = write(tmp_path / "in.jsonl", chat("x " * 118 + GH + " " + "y " * 34, "ok"))
    out = tmp_path / "out.jsonl"
    result = runner.invoke(app, ["check", str(src), "--max-seq-len", "100", "--fix", "--output", str(out)])
    assert result.exception is None or isinstance(result.exception, SystemExit)
    text = out.read_text() if out.exists() else ""
    assert GH[:20] not in text and "ghp_A1b2" not in text


# ── 4. PII and markup in role values ──────────────────────────────────────────

def test_pii_in_role_is_never_printed(tmp_path):
    src = write(tmp_path / "in.jsonl", {"messages": [{"role": EMAIL, "content": "x"},
                                                     {"role": "assistant", "content": "y"}]})
    for command in (["check"], ["check", "--json"], ["report"], ["report", "--json"], ["report", "--markdown"]):
        result = runner.invoke(app, [*command, str(src)])
        assert EMAIL not in result.output, command
    assert EMAIL not in json.dumps(check(src).to_dict())


def test_markdown_summary_keeps_role_values_inside_code(tmp_path):
    role = "x` ![p](https://attacker.invalid/p.png) [click](https://attacker.invalid) `y"
    src = write(tmp_path / "in.jsonl", {"messages": [{"role": role, "content": "x"},
                                                     {"role": "assistant", "content": "y"}]})
    md = runner.invoke(app, ["report", "--markdown", str(src)]).output
    line = next(l for l in md.splitlines() if l.startswith("- roles:"))
    assert "``" in line and "``  x`" not in line  # a longer fence wraps the value
    span = line[line.index("``"):]
    assert span.count("``") >= 2


def test_markup_in_output_path_does_not_crash(tmp_path):
    src = write(tmp_path / "in.jsonl", chat("a", "b"), chat("a", "b"))
    out = tmp_path / "out[/bold].jsonl"
    result = runner.invoke(app, ["check", str(src), "--fix", "--output", str(out)])
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert out.exists()


# ── 5. secrets the rule used to miss ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF" + "x9Kq2" * 8 + "\n-----END PGP PRIVATE KEY BLOCK-----",
    "aws_secret_access_key = " + "wJalrXUtnFEMI" + "/K7MDENG/bPxRfiCYzQ8sLpQ2mN",
    "https://hooks.slack.com/services/" + "T0A1B2C3D/B4E5F6G7H/" + "Xy9Zq8Wv7Ut6Sr5Qp4On3Ml2",
    "xoxe-" + "1-Ab3Cd5Ef7Gh9Ij2Kl4Mn6",
    "GOCSPX-" + "Ab3Cd5Ef7Gh9Ij2Kl4Mn6Op8Qr",
    "OPENAI_KEY_" + GH,
    '"SecretAccessKey": "' + "wJalrXUtnFEMI" + '/K7MDENG/bPxRfiCYzQ8sLpQ2mN"',
    "api_org_" + "Ab3Cd5Ef7Gh9Ij2Kl4Mn6Op8Qr0St1Uv2W",
])
def test_newly_detected_secrets(tmp_path, text):
    src = write(tmp_path / "in.jsonl", chat("note: " + text, "ok"))
    assert any(i["rule"] == "secrets" for i in check(src).to_dict()["issues"]), text


# ── 6. MCP server boundaries ──────────────────────────────────────────────────

def _session(tmp_path, *messages, root=None):
    import io
    wire = "".join((m if isinstance(m, str) else json.dumps(m)) + "\n" for m in messages)
    out = io.StringIO()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        serve(io.StringIO(wire), out, root=root)
    finally:
        os.chdir(cwd)
    return [json.loads(l) for l in out.getvalue().splitlines()]


def _fix(i, **arguments):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call",
            "params": {"name": "fix_dataset", "arguments": arguments}}


def test_mcp_fix_output_is_confined_and_never_clobbers(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write(work / "in.jsonl", chat("a", "b"))
    (work / "exists.jsonl").write_text("keep me\n")
    (work / "link.jsonl").symlink_to(tmp_path / "target.jsonl")
    responses = _session(
        work,
        _fix(1, path="in.jsonl", output="../escaped.jsonl"),
        _fix(2, path="in.jsonl", output=str(tmp_path / "abs.jsonl")),
        _fix(3, path="in.jsonl", output="exists.jsonl"),
        _fix(4, path="in.jsonl", output="link.jsonl", overwrite=True),
        _fix(5, path="in.jsonl", output="exists.jsonl", overwrite=True),
        _fix(6, path="in.jsonl", output="sub/new.jsonl"),
    )
    refused = [r["result"].get("isError", False) for r in responses]
    assert refused == [True, True, True, True, False, False]
    assert not (tmp_path / "escaped.jsonl").exists() and not (tmp_path / "abs.jsonl").exists()
    assert not (tmp_path / "target.jsonl").exists()
    assert (work / "exists.jsonl").read_text() != "keep me\n"
    assert (work / "sub" / "new.jsonl").exists()


def test_mcp_refuses_fifo_and_devices(tmp_path):
    fifo = tmp_path / "pipe.jsonl"
    os.mkfifo(fifo)
    responses = _session(
        tmp_path,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "check_dataset", "arguments": {"path": str(fifo)}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "check_dataset", "arguments": {"path": "/dev/zero"}}},
    )
    assert all(r["result"]["isError"] for r in responses)
    assert "regular file" in responses[0]["result"]["content"][0]["text"]


def test_mcp_survives_oversized_and_deeply_nested_lines(tmp_path):
    responses = _session(
        tmp_path,
        '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"pad":"' + "x" * 1_100_000 + '"}}',
        "[" * 100_000 + "]" * 100_000,
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    )
    assert responses[0]["error"]["code"] == -32600
    assert responses[1]["error"]["code"] == -32700
    assert responses[2] == {"jsonrpc": "2.0", "id": 3, "result": {}}


def test_mcp_relative_paths_resolve_against_root(tmp_path):
    import io
    work = tmp_path / "work"
    work.mkdir()
    write(work / "in.jsonl", chat("a", "b"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cwd = os.getcwd()
    os.chdir(elsewhere)
    out = io.StringIO()
    try:
        serve(io.StringIO(json.dumps(_fix(1, path="in.jsonl", output="out.jsonl")) + "\n"), out, root=work)
    finally:
        os.chdir(cwd)
    response = json.loads(out.getvalue())
    assert not response["result"].get("isError"), response
    assert (work / "out.jsonl").exists() and not (elsewhere / "out.jsonl").exists()


def test_mcp_refuses_kernel_pseudo_files(tmp_path):
    responses = _session(tmp_path, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "check_dataset", "arguments": {"path": "/proc/self/status"}}})
    assert responses[0]["result"]["isError"]


def test_third_pass_gaps(tmp_path):
    """Keys of message objects, ShareGPT per-turn fields, and two redacted
    keys in one object (third audit pass)."""
    GH2 = "ghp_" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"
    records = [
        {"messages": [{"role": "user", "content": "one", GH: "v"}, {"role": "assistant", "content": "ok"}]},
        {"conversations": [{"from": "human", "value": "two", "weight": GH}, {"from": "gpt", "value": "ok", "weight": 1}]},
        chat("three", "ok", metadata={GH: 1, GH2: 2}),
    ]
    src = write(tmp_path / "in.jsonl", *records)
    report = check(src).to_dict()
    assert {i["line"] for i in report["issues"] if i["rule"] == "secrets"} == {1, 2, 3}
    filtered = tmp_path / "filtered.jsonl"
    check(src, output=filtered)  # check-only output keeps only error-free records
    assert GH not in filtered.read_text()
    out = tmp_path / "out.jsonl"
    fix(src, output=out)
    text = out.read_text()
    assert GH not in text and GH2 not in text
    first, second, third = (json.loads(l) for l in text.splitlines())
    assert "[REDACTED_SECRET]" in first["messages"][0]
    assert second["conversations"][0]["weight"] == "[REDACTED_SECRET]" and second["conversations"][1]["weight"] == 1
    assert third["metadata"] == {"[REDACTED_SECRET]": 1, "[REDACTED_SECRET]_2": 2}
