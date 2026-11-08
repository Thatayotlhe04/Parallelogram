"""Offline, project-scoped MCP setup; user configuration is only printed."""
from __future__ import annotations

import json
from pathlib import Path

from .core.io import atomic_write_bytes

PROJECT_CLIENTS = {
    "claude-code": (".mcp.json", "mcpServers"),
    "cursor": (".cursor/mcp.json", "mcpServers"),
    "vscode": (".vscode/mcp.json", "servers"),
    "gemini": (".gemini/settings.json", "mcpServers"),
}
MANUAL_CLIENTS = {
    "codex": "~/.codex/config.toml",
    "windsurf": "~/.codeium/windsurf/mcp_config.json",
    "cline": "cline_mcp_settings.json (Cline > MCP Servers > Configure > Configure MCP Servers)",
    "zed": "~/.config/zed/settings.json (Zed: Open Settings File)",
}
START = "<!-- parallelogram:start -->"
END = "<!-- parallelogram:end -->"
# Kept identical to the pasteable block in docs/agents.md (tested).
AGENT_INSTRUCTIONS = """Run Parallelogram before fine-tuning and after every dataset edit.
Prefer: parallelogram check train.jsonl --json --limit 50.
Check exits: 0 no errors/warnings, 1 warnings, 2 errors/invalid arguments, 3 operational failure.
Fix exits: 0 all emitted, 1 partial fix, 2 none emitted; report exit 3 means baseline regression.
Inspect exact summary/by_rule counts and issues_truncated; use rules --json for repair guidance.
Repair with --fix --output clean.jsonl, never overwrite input, then recheck the output."""


def server_config(client: str) -> dict:
    config = {"command": "parallelogram", "args": ["mcp"]}
    if client == "vscode":
        config = {"type": "stdio", **config}
    return config


def manual_setup(client: str) -> str:
    destination = MANUAL_CLIENTS[client]
    if client == "codex":
        return (f"Run: codex mcp add parallelogram -- parallelogram mcp\n"
                f"Or add to {destination}:\n\n"
                '[mcp_servers.parallelogram]\ncommand = "parallelogram"\nargs = ["mcp"]\n')
    key = "context_servers" if client == "zed" else "mcpServers"
    extra = ""
    if client == "cline":
        extra = "\nCline CLI uses ~/.cline/data/settings/cline_mcp_settings.json."
    if client == "windsurf":
        extra = ("\nFor Windsurf installations: Settings > Manage MCPs > View raw config."
                 "\nNewer Devin Desktop uses different settings; consult its MCP configuration.")
    return (f"Merge into {destination}{extra}:\n\n"
            + json.dumps({key: {"parallelogram": server_config(client)}}, indent=2) + "\n")


def _project_path(root: Path, relative: str) -> Path:
    path = root / relative
    if root == Path.home().resolve():
        raise ValueError("Run integrate inside a project directory, not your home directory.")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Refusing project path outside the project (symlink): {relative}")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _instructions(existing: str) -> str:
    block = f"{START}\n{AGENT_INSTRUCTIONS}\n{END}"
    if START in existing or END in existing:
        if existing.count(START) != 1 or existing.count(END) != 1 or existing.index(END) < existing.index(START):
            raise ValueError("Agent instruction markers are malformed; repair them before integrating.")
        start, end = existing.index(START), existing.index(END) + len(END)
        return existing[:start] + block + existing[end:]
    separator = "\n\n" if existing and not existing.endswith("\n") else "\n" if existing else ""
    return existing + separator + block + "\n"


def integrate(client: str, *, root: Path, agents_md: bool = False, dry_run: bool = False) -> list[str]:
    """Plan every edit before writing; no user-level config is ever modified."""
    if client not in PROJECT_CLIENTS and client not in MANUAL_CLIENTS:
        raise ValueError(f"Unknown client {client!r}. Use parallelogram integrate --list.")
    root = root.resolve()
    plans: list[tuple[Path, str, str]] = []
    messages: list[str] = []
    if client in PROJECT_CLIENTS:
        relative, key = PROJECT_CLIENTS[client]
        path = _project_path(root, relative)
        before = _read(path)
        try:
            config = json.loads(before) if path.exists() else {}
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"{relative} is not valid JSON; refusing to overwrite it.") from exc
        if not isinstance(config, dict) or not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{relative} must contain a JSON object with an object-valued {key}.")
        servers = config.setdefault(key, {})
        current = servers.get("parallelogram", {})
        if not isinstance(current, dict):
            raise ValueError(f"{relative}: the parallelogram server must be an object.")
        desired = {**current, **server_config(client)}
        if servers.get("parallelogram") == desired:
            after = before  # preserve formatting and mtime on repeat runs
        else:
            servers["parallelogram"] = desired
            after = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        plans.append((path, before, after))
    else:
        messages.append("MCP configuration is manual; no client configuration written.\n" + manual_setup(client))
    if agents_md:
        name = "CLAUDE.md" if client == "claude-code" else "AGENTS.md"
        path = _project_path(root, name)
        before = _read(path)
        plans.append((path, before, _instructions(before)))
    for path, before, after in plans:
        relative = path.relative_to(root)
        if before == after:
            messages.append(f"Unchanged: {relative}")
        elif dry_run:
            messages.append(f"Would write {relative}:\n{after}")
        else:
            atomic_write_bytes(path, after.encode("utf-8"))
            messages.append(f"Wrote {relative}")
    return messages
