"""Minimal offline MCP tool server; stdout is exclusively JSON-RPC messages."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from contextlib import redirect_stdout
from typing import TextIO

from . import __version__, api
from .core.report import scrub
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
        "description": "Mechanically repair a local fine-tuning JSONL dataset into the required output path, preserving each record's source format. Redacts detected secrets/PII, repairs encoding, truncates overlong user turns, and drops duplicates, eval overlap and unrepairable records. Returns the fix JSON plus exit_code: 0 every record was emitted, 1 some were dropped or unparseable, 2 nothing was emitted. summary and fixes_by_rule are exact; outcomes lists only records that were dropped, unparseable or fixed (in that order, then by line), at most limit of them (default 50), with outcomes_truncated. Review dropped records for data loss; use dry_run=true to preview without writing. Relative paths are resolved against the server's root directory (its working directory unless started with --root). Output must be inside that root, must differ from the input and eval files, and must not already exist unless overwrite=true. Recheck the repaired copy with check_dataset before training.",
        "inputSchema": _schema({**_COMMON,
                                "output": {"type": "string", "minLength": 1},
                                "dry_run": {"type": "boolean", "default": False},
                                "overwrite": {"type": "boolean", "default": False,
                                              "description": "Replace an existing output file. Without it, an existing output is refused."},
                                "limit": {"type": "integer", "minimum": 0, "default": 50}}, ["path", "output"]),
    },
    {
        "name": "list_rules",
        "description": "List every registered validation rule with its id, default severity, fixable flag, fix_scope, description and fix_description. Use this metadata to interpret check_dataset diagnostics and decide which mechanical repairs are appropriate. Context-window findings can be warnings when token counts are estimated; eval-overlap runs only when eval_path is supplied, and schema cannot be disabled.",
        "inputSchema": _schema({}, []),
    },
]


# Served only with `parallelogram mcp --optimize`, so the default tool list
# (and every client configured against it) is unchanged. There is no
# optimize_run tool on purpose: a run trains models for minutes to hours, and
# this server answers one request at a time. Start runs from the CLI.
OPTIMIZE_TOOLS = [
    {
        "name": "optimize_plan",
        "description": "Plan a Parallelogram Optimize experiment without training. Reads the experiment contract (.yaml/.yml/.json), checks the candidate, development and test splits for validation errors and leakage, and returns the experiment matrix, estimated compute by stage against the budget, what does not fit, resource assumptions and the artifacts a run would write. exit_code 0 means the plan fits its budget, 2 that it does not. Token counts are exact only when the optional ML extras and the tokenizer are available locally; otherwise they are labeled as estimates. Nothing is downloaded and nothing is written.",
        "inputSchema": _schema({"experiment": {"type": "string", "minLength": 1,
                                               "description": "Path to the experiment contract."}}, ["experiment"]),
    },
    {
        "name": "optimize_report",
        "description": "Report on an optimize run directory, rebuilt from its artifacts. Sections are kept apart: estimated_signals (gradient-alignment heuristics, not probabilities), development_results (measured, but they chose the candidate), final_results (measured once on held-out data after the freeze), compute (everything charged, including scoring overhead), coverage, limitations and not_performed. conclusion.status is supported, not_supported or insufficient_evidence; results labeled pilot do not establish an improvement for larger models. Read conclusion and not_performed before quoting any number.",
        "inputSchema": _schema({"run_dir": {"type": "string", "minLength": 1,
                                            "description": "Run directory written by `parallelogram optimize run`."}},
                               ["run_dir"]),
    },
    {
        "name": "optimize_export",
        "description": "Copy the records selected by an optimize run to a new JSONL file, byte for byte in source order, and write <output>.manifest.json with source identifiers, selection reasons, scores, fingerprints and provenance. Exports the frozen candidate unless trial names another arm. Refuses to write over an input dataset, outside the server root, or over an existing file unless overwrite=true, and refuses if the candidates file changed since the run. Returns the manifest without its per-record list (records_listed=false); read the manifest file for per-record detail.",
        "inputSchema": _schema({"run_dir": {"type": "string", "minLength": 1},
                                "output": {"type": "string", "minLength": 1},
                                "trial": {"type": "string", "minLength": 1,
                                          "description": "Arm to export instead of the frozen candidate."},
                                "overwrite": {"type": "boolean", "default": False}}, ["run_dir", "output"]),
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


def _arguments(name: str, arguments: object, tools: list[dict] = TOOLS) -> dict:
    tool = next((tool for tool in tools if tool["name"] == name), None)
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


class ToolRefused(Exception):
    """A tool call the server won't carry out; returned as an isError result."""


_PSEUDO_FS = ("/proc", "/sys", "/dev")


def _regular_file(value: str, name: str) -> None:
    # A FIFO or a device (/dev/stdin, /dev/zero) would block or never end and
    # the server handles one request at a time, so only regular files are read.
    # Kernel pseudo-files (/proc/kmsg) report as regular files but can block too.
    resolved = str(Path(value).resolve())
    if not Path(value).is_file() or any(
            resolved == top or resolved.startswith(top + "/") for top in _PSEUDO_FS):
        raise ToolRefused(f"{name} is not a regular file: {value}")


def _confined_output(value: str, root: Path, overwrite: bool, dry_run: bool) -> None:
    out = Path(value)
    if out.is_symlink():
        raise ToolRefused(f"output is a symlink; refusing to write through it: {value}")
    resolved = out.resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolRefused(f"output must be inside {root}: {value}")
    if out.exists() and not overwrite and not dry_run:
        raise ToolRefused(f"output already exists; pass overwrite=true to replace it: {value}")


def _optimize_tool(name: str, arguments: dict, root: Path) -> dict:
    """Run one optimize tool. Imported lazily: the default server never loads it."""
    from . import optimize

    if name == "optimize_plan":
        _regular_file(arguments["experiment"], "experiment")
        obj = optimize.plan(arguments["experiment"])
        return {**obj, "exit_code": 0 if obj["fits_budget"] else 2}
    if not Path(arguments["run_dir"]).is_dir():
        raise ToolRefused(f"run_dir is not a directory: {arguments['run_dir']}")
    if name == "optimize_report":
        return {**optimize.report(arguments["run_dir"]), "exit_code": 0}
    overwrite = arguments.get("overwrite", False)
    for target in (arguments["output"], arguments["output"] + ".manifest.json"):
        _confined_output(target, root, overwrite, False)
    manifest = optimize.export(arguments["run_dir"], arguments["output"], trial=arguments.get("trial"),
                               overwrite=overwrite)
    return {**{k: v for k, v in manifest.items() if k != "records"}, "records_listed": False, "exit_code": 0}


def _call_tool(params: dict, root: Path | None = None, optimize: bool = False) -> dict:
    if not isinstance(params.get("name"), str):
        raise InvalidParams("tools/call requires a string name")
    name = params["name"]
    arguments = _arguments(name, params.get("arguments", {}), TOOLS + OPTIMIZE_TOOLS if optimize else TOOLS)
    root = (root or Path(os.getcwd())).resolve()
    try:
        if name.startswith("optimize_"):
            obj = _optimize_tool(name, arguments, root)
            return {"content": [{"type": "text", "text": json.dumps(obj, separators=(",", ":"))}],
                    "structuredContent": obj}
        if name in ("check_dataset", "fix_dataset"):
            _regular_file(arguments["path"], "path")
            if "eval_path" in arguments:
                _regular_file(arguments["eval_path"], "eval_path")
        if name == "fix_dataset":
            _confined_output(arguments["output"], root, arguments.pop("overwrite", False),
                             arguments.get("dry_run", False))
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
        return {"isError": True, "content": [{"type": "text", "text": scrub(str(exc))}]}


def _error(request_id: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: object, root: Path | None = None, optimize: bool = False) -> dict | None:
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
            result = {"tools": TOOLS + OPTIMIZE_TOOLS if optimize else TOOLS}
        elif method == "tools/call":
            result = _call_tool(params, root, optimize)
        else:
            return _error(request_id, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except InvalidParams as exc:
        return _error(request_id, -32602, str(exc))


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


MAX_LINE = 1_000_000  # characters; real requests are a few hundred


def _lines(stdin: TextIO):
    """Yield request lines, or None for one longer than MAX_LINE.

    Reading in bounded chunks keeps a client that never sends a newline from
    growing the server's memory without limit.
    """
    while True:
        line = stdin.readline(MAX_LINE + 1)
        if not line:
            return
        if len(line) > MAX_LINE and not line.endswith("\n"):
            while True:  # discard the rest of the oversized line
                more = stdin.readline(MAX_LINE + 1)
                if not more or more.endswith("\n"):
                    break
            yield None
            continue
        yield line


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None,
          root: Path | None = None, optimize: bool = False) -> None:
    """Process one JSON object per line until EOF, flushing every response."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    root = Path(root or os.getcwd()).resolve()
    # Relative paths in tool calls resolve against the root. This process
    # serves one client, so changing its working directory is safe.
    os.chdir(root)
    for line in _lines(stdin):
        if line is None:
            response = _error(None, -32600, "Request too large")
        else:
            try:
                message = json.loads(line, parse_constant=_invalid_constant)
            except (ValueError, RecursionError):
                response = _error(None, -32700, "Parse error")
            else:
                # Keep accidental downstream diagnostics off the transport.
                with redirect_stdout(sys.stderr):
                    try:
                        response = handle(message, root, optimize)
                    except Exception as exc:  # noqa: BLE001 — isolate tool/transport failures
                        print(scrub(f"MCP internal error: {type(exc).__name__}: {exc}"), file=sys.stderr)
                        response = _error(message.get("id") if isinstance(message, dict) else None,
                                          -32603, "Internal error")
        if response is not None:
            stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
