"""Exercise the actual stdio process, including recovery and clean EOF."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from parallelogram import __version__, check, fix, rules
from parallelogram.mcp_server import PROTOCOL_VERSIONS, handle, serve


def request(request_id, method, params=None):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def tool(request_id, name, arguments=None):
    return request(request_id, "tools/call", {"name": name, "arguments": arguments or {}})


def session(tmp_path, messages):
    wire = "\n".join(m if isinstance(m, str) else json.dumps(m) for m in messages) + "\n"
    process = subprocess.run([sys.executable, "-m", "parallelogram.cli", "mcp"],
                             input=wire, capture_output=True, text=True, cwd=tmp_path, timeout=20)
    assert process.returncode == 0, process.stderr
    assert process.stderr == ""
    return [json.loads(line) for line in process.stdout.splitlines()]


def unpack(response):
    result = response["result"]
    assert not result.get("isError", False)
    obj = result["structuredContent"]
    assert json.loads(result["content"][0]["text"]) == obj
    assert result["content"][0]["type"] == "text"
    return obj


def test_real_mcp_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "train.jsonl"
    path.write_text('{"prompt":"Q","completion":"A"}\n' * 60)
    before = path.read_bytes()
    messages = [
        request(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "pytest", "version": "1"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request(2, "ping"), request(3, "tools/list"),
        tool(4, "check_dataset", {"path": "train.jsonl"}),
        tool(5, "fix_dataset", {"path": "train.jsonl", "output": "clean.jsonl"}),
        tool(6, "list_rules"), request(7, "unknown"),
        '{bad json', request(8, "ping"),
        tool(9, "fix_dataset", {"path": "train.jsonl", "output": "preview.jsonl", "dry_run": True}),
    ]
    responses = session(tmp_path, messages)
    assert len(responses) == 10  # no initialized response
    assert responses[0]["result"] == {
        "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
        "serverInfo": {"name": "parallelogram", "version": __version__}}
    assert responses[1]["result"] == {}
    advertised = responses[2]["result"]["tools"]
    assert {t["name"] for t in advertised} == {"check_dataset", "fix_dataset", "list_rules"}
    assert all(t["description"] and t["inputSchema"]["type"] == "object" for t in advertised)
    checked = unpack(responses[3])
    assert checked.pop("exit_code") == 2
    assert checked == check("train.jsonl", limit=50).to_dict()
    assert len(checked["issues"]) == 50 and checked["issues_truncated"]
    assert checked["summary"]["errors"] == 59
    fixed, full = unpack(responses[4]), fix("train.jsonl").to_dict()
    assert fixed.pop("exit_code") == 1
    assert fixed.pop("outcomes_truncated") is True
    assert fixed.pop("outcomes") == [o for o in full.pop("outcomes") if o["disposition"] != "unchanged"][:50]
    assert fixed == full
    assert unpack(responses[5]) == {"rules": rules()}
    assert responses[6]["error"]["code"] == -32601
    assert responses[7] == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    assert responses[8]["result"] == {}
    assert unpack(responses[9])["dry_run"] is True
    assert not (tmp_path / "preview.jsonl").exists()
    assert path.read_bytes() == before
    assert check("clean.jsonl").exit_code == 0
    assert {p.name for p in tmp_path.iterdir()} == {"train.jsonl", "clean.jsonl"}


@pytest.mark.parametrize("version", [*PROTOCOL_VERSIONS, "unknown", None])
def test_protocol_negotiation(tmp_path, version):
    params = {} if version is None else {"protocolVersion": version}
    result = session(tmp_path, [request("init", "initialize", params)])[0]
    assert result["id"] == "init"
    assert result["result"]["protocolVersion"] == (version if version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0])


@pytest.mark.parametrize("message", [
    request(1, "tools/call"),
    request(1, "tools/call", {"name": 1}),
    tool(1, "missing_tool"),
    tool(1, "check_dataset"),
    tool(1, "fix_dataset", {"path": "data"}),
    tool(1, "check_dataset", {"path": 1}),
    tool(1, "check_dataset", {"path": ""}),
    tool(1, "check_dataset", {"path": "data", "limit": -1}),
    tool(1, "check_dataset", {"path": "data", "limit": True}),
    tool(1, "check_dataset", {"path": "data", "limit": 1.5}),
    tool(1, "check_dataset", {"path": "data", "limit": None}),
    tool(1, "check_dataset", {"path": "data", "disable": "pii"}),
    tool(1, "check_dataset", {"path": "data", "disable": [False]}),
    tool(1, "check_dataset", {"path": "data", "format": "unknown"}),
    tool(1, "check_dataset", {"path": "data", "other": 2}),
    tool(1, "fix_dataset", {"path": "data", "output": "out", "dry_run": "false"}),
    tool(1, "list_rules", {"extra": "bad"}),
    request(1, "tools/call", {"name": "list_rules", "arguments": []}),
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": []},
    request(1, "initialize", {"protocolVersion": 5}),
])
def test_bad_params_are_rpc_errors(message):
    assert handle(message)["error"]["code"] == -32602


def test_tool_failures_are_results_and_server_continues(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"prompt":"Q","completion":"A"}\n')
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(path)
    calls = [
        tool(1, "check_dataset", {"path": "missing.jsonl"}),
        tool(2, "fix_dataset", {"path": "train.jsonl", "output": "train.jsonl"}),
        tool(3, "fix_dataset", {"path": "train.jsonl", "output": "alias.jsonl", "dry_run": True}),
        tool(4, "check_dataset", {"path": "train.jsonl", "disable": ["schema"]}),
        tool(5, "check_dataset", {"path": "train.jsonl", "disable": ["typo"]}),
        tool(6, "fix_dataset", {"path": "train.jsonl", "output": "."}),
        request(7, "ping"),
    ]
    responses = session(tmp_path, calls)
    for response in responses[:-1]:
        assert "error" not in response
        assert response["result"]["isError"] is True
        assert response["result"]["content"][0]["text"]
    assert responses[-1]["result"] == {}
    assert path.read_text() == '{"prompt":"Q","completion":"A"}\n'


def test_eval_disable_and_format_pass_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train").write_text('{"prompt":"Q","completion":"A"}\n')
    (tmp_path / "eval").write_text('{"prompt":"Q","completion":"Different"}\n')
    args = {"path": "train", "eval_path": "eval", "disable": ["pii"], "format": "prompt-completion"}
    responses = session(tmp_path, [tool(1, "check_dataset", {**args, "limit": 0}),
                                   tool(2, "fix_dataset", {**args, "output": "out"})])
    checked = check(**args, limit=0)
    assert unpack(responses[0]) == {**checked.to_dict(), "exit_code": checked.exit_code}
    fixed = fix(**args)
    full = fixed.to_dict()
    assert unpack(responses[1]) == {**full, "outcomes": [o for o in full["outcomes"] if o["disposition"] != "unchanged"],
                                    "outcomes_truncated": False, "exit_code": fixed.exit_code}
    assert (tmp_path / "out").read_bytes() == b""
    assert (tmp_path / "eval").read_text() == '{"prompt":"Q","completion":"Different"}\n'


@pytest.mark.parametrize("message", ["null", "[]", "1", '{}',
                                    '{"jsonrpc":"1.0","method":"ping","id":1}',
                                    '{"jsonrpc":"2.0","method":"ping","id":true}'])
def test_invalid_requests_survive(tmp_path, message):
    responses = session(tmp_path, [message, request(3, "ping")])
    assert responses[0]["error"]["code"] == -32600
    assert responses[1]["result"] == {}


def test_non_json_constants_are_parse_errors(tmp_path):
    responses = session(tmp_path, ['{"jsonrpc":"2.0","id":NaN,"method":"ping"}', request(1, "ping")])
    assert responses[0]["error"]["code"] == -32700
    assert responses[1]["result"] == {}


def test_stdout_diagnostics_redirected(monkeypatch, capsys):
    from io import StringIO
    from parallelogram import api

    def noisy_rules():
        print("downstream diagnostic")
        return []

    monkeypatch.setattr(api, "rules", noisy_rules)
    output = StringIO()
    serve(StringIO(json.dumps(tool(1, "list_rules")) + "\n"), output)
    assert unpack(json.loads(output.getvalue())) == {"rules": []}
    assert capsys.readouterr() == ("", "downstream diagnostic\n")


def test_eof_without_messages_exits_cleanly(tmp_path):
    process = subprocess.run([sys.executable, "-m", "parallelogram.cli", "mcp"],
                             input="", capture_output=True, text=True, cwd=tmp_path, timeout=10)
    assert process.returncode == 0
    assert process.stdout == process.stderr == ""


def test_fix_result_stays_agent_sized(tmp_path):
    """A large clean dataset must not return one outcome per record."""
    rows = [json.dumps({"prompt": f"Q{i}", "completion": f"A{i}"}) for i in range(3000)]
    rows[10] = rows[0]                       # one duplicate -> dropped
    rows[20] = "not json"                    # one unparseable line
    rows[30] = json.dumps({"prompt": "Mail me at someone@gmail.com", "completion": "OK"})  # fixed
    (tmp_path / "train.jsonl").write_text("\n".join(rows) + "\n")
    responses = session(tmp_path, [tool(1, "fix_dataset", {"path": "train.jsonl", "output": "clean.jsonl"}),
                                   tool(2, "fix_dataset", {"path": "train.jsonl", "output": "c2.jsonl", "limit": 1}),
                                   tool(3, "check_dataset", {"path": "clean.jsonl"})])
    first = unpack(responses[0])
    assert first["summary"]["total"] == 3000 and first["summary"]["unchanged"] == 2997
    assert [o["disposition"] for o in first["outcomes"]] == ["dropped", "unparseable", "fixed"]
    assert first["outcomes_truncated"] is False and first["exit_code"] == 1
    assert len(json.dumps(responses[0])) < 4000
    second = unpack(responses[1])
    assert [o["line"] for o in second["outcomes"]] == [11] and second["outcomes_truncated"] is True
    assert unpack(responses[2])["exit_code"] == 0
