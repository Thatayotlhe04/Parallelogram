"""Minimal offline MCP tool server; stdout is exclusively JSON-RPC messages."""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import TextIO

from . import __version__, api
from .formats import supported_formats

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

_COMMON = {
    "path": {"type": "string", "minLength": 1, "description": "Dataset path, relative to the server working directory or absolute."},
    "format": {"type": "string", "enum": supported_formats(), "default": "auto"},
    "eval_path": {"type": "string", "minLength": 1, "description": "Optional eval dataset for train/eval overlap checks."},
    "disable": {"type": "array", "items": {"type": "string"}, "default": [],
                "description": "Rule ids to disable; schema cannot be disabled. Disabling rules voids the full validation guarantee."},
}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


TOOLS = [
    {
        "name": "check_dataset",
        "description": "Validate a local fine-tuning JSONL dataset without changing files. Accepts auto-detected chat, ShareGPT, prompt/completion, Alpaca and preference formats; eval_path also checks train/eval leakage. Returns schema_version 1 check JSON plus exit_code: 0 means no errors or warnings, 1 warnings only, 2 errors. Judge the result by exit_code, not summary.clean, which is also false when there is only an informational note. Counts are exact; at most limit issues (default 50) are listed, errors first, then warnings, then by line. Inspect summary and by_rule even when issues_truncated is true; use list_rules to understand repairs and fix_dataset to write a separate repaired copy.",
        "inputSchema": _schema({**_COMMON, "limit": {"type": "integer", "minimum": 0, "default": 50}}, ["path"]),
    },
    {
        "name": "fix_dataset",
        "description": "Mechanically repair a local fine-tuning JSONL dataset into the required output path, preserving each record's source format. Redacts detected secrets/PII, repairs encoding, truncates overlong user turns, and drops duplicates, eval overlap and unrepairable records. Returns the fix JSON plus exit_code: 0 every record was emitted, 1 some were dropped or unparseable, 2 nothing was emitted. summary and fixes_by_rule are exact; outcomes lists only records that were dropped, unparseable or fixed (in that order, then by line), at most limit of them (default 50), with outcomes_truncated. Review dropped records for data loss; use dry_run=true to preview without writing. Output must differ from input and eval paths. Recheck the repaired copy with check_dataset before training.",
        "inputSchema": _schema({**_COMMON,
                                "output": {"type": "string", "minLength": 1},
                                "dry_run": {"type": "boolean", "default": False},
                                "limit": {"type": "integer", "minimum": 0, "default": 50}}, ["path", "output"]),
    },
    {
        "name": "list_rules",
        "description": "List every registered validation rule with its id, default severity, fixable flag, fix_scope, description and fix_description. Use this metadata to interpret check_dataset diagnostics and decide which mechanical repairs are appropriate. Context-window findings can be warnings when token counts are estimated; eval-overlap runs only when eval_path is supplied, and schema cannot be disabled.",
        "inputSchema": _schema({}, []),
    },
]


_OUTCOME_ORDER = {"dropped": 0, "unparseable": 1, "fixed": 2}


def _bounded_fix(obj: dict, limit: int) -> dict:
    """Keep a fix result agent-sized: unchanged records are only counted, and
    the records that changed are listed data loss first, at most `limit`."""
    changed = [o for o in obj.get("outcomes", []) if o.get("disposition") in _OUTCOME_ORDER]
    changed.sort(key=lambda o: (_OUTCOME_ORDER[o["disposition"]], o.get("line") or 0))
    return {**obj, "outcomes": changed[:limit], "outcomes_truncated": len(changed) > limit}


class InvalidParams(ValueError):
    """Malformed method arguments, distinct from failures while using a tool."""


def _arguments(name: str, arguments: object) -> dict:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        raise InvalidParams(f"Unknown tool: {name}")
    schema = tool["inputSchema"]
    if not isinstance(arguments, dict):
        raise InvalidParams("arguments must be an object")
    missing = set(schema["required"]) - arguments.keys()
    extra = arguments.keys() - schema["properties"].keys()
    if missing or extra:
        raise InvalidParams(f"Missing arguments: {sorted(missing)}; unknown arguments: {sorted(extra)}")
    for key, value in arguments.items():
        spec = schema["properties"][key]
        kind = spec["type"]
        valid = ((kind == "string" and isinstance(value, str))
                 or (kind == "integer" and type(value) is int)
                 or (kind == "boolean" and type(value) is bool)
                 or (kind == "array" and isinstance(value, list)
                     and all(isinstance(item, str) for item in value)))
        if not valid:
            raise InvalidParams(f"{key} must have type {kind}")
        if "minLength" in spec and len(value) < spec["minLength"]:
            raise InvalidParams(f"{key} must not be empty")
        if "minimum" in spec and value < spec["minimum"]:
            raise InvalidParams(f"{key} must be at least {spec['minimum']}")
        if "enum" in spec and value not in spec["enum"]:
            raise InvalidParams(f"{key} must be one of {spec['enum']}")
    return arguments


def _call_tool(params: dict) -> dict:
    if not isinstance(params.get("name"), str):
        raise InvalidParams("tools/call requires a string name")
    name = params["name"]
    arguments = _arguments(name, params.get("arguments", {}))
    try:
        if name == "check_dataset":
            result = api.check(**{"limit": 50, **arguments})
            obj = {**result.to_dict(), "exit_code": result.exit_code}
        elif name == "fix_dataset":
            limit = arguments.pop("limit", 50)
            result = api.fix(**arguments)
            obj = _bounded_fix(result.to_dict(), limit)
            obj["exit_code"] = result.exit_code
        else:
            # structuredContent must be an object, so wrap the API's list.
            obj = {"rules": api.rules()}
        return {"content": [{"type": "text", "text": json.dumps(obj, separators=(",", ":"))}],
                "structuredContent": obj}
    except Exception as exc:  # noqa: BLE001 — isolate tool/transport failures
        return {"isError": True, "content": [{"type": "text", "text": str(exc)}]}


def _error(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: object) -> dict | None:
    """Dispatch one JSON-RPC message without writing to either stream."""
    if (not isinstance(message, dict) or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
            or ("id" in message and message["id"] is not None
                and type(message["id"]) not in (str, int))):
        return _error(None, -32600, "Invalid Request")
    # Notifications, including initialized/cancelled, never elicit a response.
    if "id" not in message:
        return None
    request_id = message["id"]
    method = message["method"]
    params = message.get("params", {})
    try:
        if not isinstance(params, dict):
            raise InvalidParams("params must be an object")
        if method == "initialize":
            version = params.get("protocolVersion")
            if version is not None and not isinstance(version, str):
                raise InvalidParams("protocolVersion must be a string")
            result = {"protocolVersion": version if version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "parallelogram", "version": __version__}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = _call_tool(params)
        else:
            return _error(request_id, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except InvalidParams as exc:
        return _error(request_id, -32602, str(exc))


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Process one JSON object per line until EOF, flushing every response."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    for line in stdin:
        try:
            message = json.loads(line, parse_constant=_invalid_constant)
        except ValueError:
            response = _error(None, -32700, "Parse error")
        else:
            # Keep accidental downstream diagnostics off the transport.
            with redirect_stdout(sys.stderr):
                try:
                    response = handle(message)
                except Exception as exc:  # noqa: BLE001 — isolate tool/transport failures
                    print(f"MCP internal error: {exc}", file=sys.stderr)
                    response = _error(message.get("id") if isinstance(message, dict) else None,
                                      -32603, "Internal error")
        if response is not None:
            stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
