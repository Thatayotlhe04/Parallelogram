from __future__ import annotations

import json
from pathlib import Path

import pytest
from parallelogram.cli import app
from parallelogram.integrations import (
    AGENT_INSTRUCTIONS,
    END,
    MANUAL_CLIENTS,
    PROJECT_CLIENTS,
    START,
)
from typer.testing import CliRunner


def invoke(*args):
    return CliRunner().invoke(app, ["integrate", *args])


@pytest.mark.parametrize("client", PROJECT_CLIENTS)
def test_project_merge_and_idempotence(tmp_path, monkeypatch, client):
    monkeypatch.chdir(tmp_path)
    relative, key = PROJECT_CLIENTS[client]
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"other": [1, 2], key: {"existing": {"command": "keep"}, "parallelogram": {"env": {"KEEP": "value"}}}}
    path.write_text(json.dumps(config))
    assert invoke(client, "--agents-md").exit_code == 0
    actual = json.loads(path.read_text())
    assert actual["other"] == config["other"]
    assert actual[key]["existing"] == config[key]["existing"]
    assert actual[key]["parallelogram"] == {"env": {"KEEP": "value"}, "command": "parallelogram", "args": ["mcp"], **({"type": "stdio"} if client == "vscode" else {})}
    instructions = tmp_path / ("CLAUDE.md" if client == "claude-code" else "AGENTS.md")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, instructions)}
    assert invoke(client, "--agents-md").exit_code == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert instructions.read_text().count(START) == 1


@pytest.mark.parametrize("client", [*PROJECT_CLIENTS, *MANUAL_CLIENTS])
def test_dry_run(tmp_path, monkeypatch, client):
    monkeypatch.chdir(tmp_path)
    result = invoke(client, "--agents-md", "--dry-run")
    assert result.exit_code == 0 and "Would write" in result.output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("client", MANUAL_CLIENTS)
def test_manual_only(tmp_path, monkeypatch, client):
    monkeypatch.chdir(tmp_path)
    result = invoke(client)
    assert result.exit_code == 0
    assert MANUAL_CLIENTS[client] in result.output
    assert "parallelogram" in result.output and "mcp" in result.output
    assert list(tmp_path.iterdir()) == []
    assert invoke(client, "--agents-md").exit_code == 0
    assert [p.name for p in tmp_path.iterdir()] == ["AGENTS.md"]


@pytest.mark.parametrize("content", ["", "{broken", "[]", '{"mcpServers":[]}', '{"mcpServers":{"parallelogram":false}}'])
def test_invalid_json_not_overwritten(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".mcp.json"
    path.write_text(content)
    result = invoke("claude-code", "--agents-md")
    assert result.exit_code == 2
    assert path.read_text() == content
    assert not (tmp_path / "CLAUDE.md").exists()


def test_markers_replace_only_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "AGENTS.md"
    path.write_text(f"Before\n{START}\nold\n{END}\nAfter\n")
    assert invoke("codex", "--agents-md").exit_code == 0
    assert path.read_text() == f"Before\n{START}\n{AGENT_INSTRUCTIONS}\n{END}\nAfter\n"


def test_malformed_markers_prevent_all_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text(START)
    assert invoke("cursor", "--agents-md").exit_code == 2
    assert not (tmp_path / ".cursor").exists()


def test_symlinks_cannot_write_outside_project(tmp_path, monkeypatch):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / ".cursor").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(project)
    assert invoke("cursor").exit_code == 2
    assert not list(outside.iterdir())


def test_home_not_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert invoke("gemini").exit_code == 2
    assert list(tmp_path.iterdir()) == []


def test_list_and_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke("--list")
    assert result.exit_code == 0
    assert all(c in result.output for c in [*PROJECT_CLIENTS, *MANUAL_CLIENTS])
    assert invoke("unknown").exit_code == 2
    assert invoke().exit_code == 2
    assert list(tmp_path.iterdir()) == []


def test_documented_block_matches_package():
    docs = (Path(__file__).parents[1] / "docs" / "agents.md").read_text()
    assert f"```text\n{AGENT_INSTRUCTIONS}\n```" in docs
