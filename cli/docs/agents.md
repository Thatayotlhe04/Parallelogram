# Parallelogram for agents

Parallelogram 1.2 validates and mechanically repairs local fine-tuning JSONL.
Use the Python API, JSON CLI, or stdio MCP tools before training and after editing
data. All three share the same validation and fix paths. Runtime dependencies
remain `typer` and `rich`; MCP needs no extra package, service, or network.

```bash
pip install parallelogram
parallelogram check train.jsonl --json --limit 50
parallelogram rules --json
parallelogram check train.jsonl --fix --dry-run --json
parallelogram check train.jsonl --fix --output clean.jsonl --json
parallelogram check clean.jsonl --json --limit 50
```

## Python API

```python
from parallelogram import ParallelogramError, check, fix, rules

result = check(
    "train.jsonl", format="auto", eval_path=None, disable=(),
    tokenizer=None, max_seq_len=4096, limit=50,
)
status = result.exit_code
payload = result.to_dict()  # identical to check --json --limit 50

preview = fix("train.jsonl", output="clean.jsonl", dry_run=True)
fixed = fix("train.jsonl", output="clean.jsonl", dry_run=False)
fix_payload = fixed.to_dict()  # identical to check --fix --output clean.jsonl --json
fix_status = fixed.exit_code
metadata = rules()  # list of rule-info dicts, also parallelogram rules --json
```

Paths accept strings or `pathlib.Path`. Optional arguments are keyword-only.
`fix` accepts the same format, eval_path, disable, tokenizer and max_seq_len
options as `check`; it does not accept `limit`. Omitting output returns a fix
projection without writing. `check(..., output="filtered.jsonl")` copies only
error-free source lines verbatim, including any unrepaired warnings.

The API prints nothing. Data findings return a result; they do not raise.
Invalid options and output paths aliasing the input or eval file raise
`ValueError`. File read/write failures raise `ParallelogramError` with a message.
CLI check/fix maps these operational failures to exit 3; invalid command-line
arguments (including missing paths rejected during argument parsing) exit 2.
Fixer-rule exceptions remain visible in `fixer_errors` and retain the existing
partial-fix exit semantics.

Output must differ from input and eval, including resolved relative paths,
symlinks and hard links. This applies even to dry runs. A dry run writes nothing;
a real write atomically replaces only the requested output file. Existing output
files may be replaced. Source formats are preserved, including mixed-format files.

## Reading results cheaply

The [JSON contract](json-output.md) stays at `schema_version: 1`.
`limit` bounds only the serialized issue list; it does not stop validation or
change any counts, output records, or exit status. `None` (the API/CLI default)
includes all issues; zero returns counts only. MCP defaults to 50 issues.

Issues sort by error, warning, then info, and by ascending physical line within
each severity. Dataset-wide notes with a null line sort last within their
severity; ties retain validation order. `issues_truncated` says whether any
issues were omitted. `by_rule` maps each rule that emitted issues to exact
`errors`, `warnings`, and `info` counts, including issues hidden by the limit.
Use summary/by_rule to decide whether the dataset is usable; an empty issues
array with limit zero does not mean the dataset passed.

`--limit` applies to check JSON diagnostics, not terminal rendering or fix
outcomes. Combining it with `--fix` is rejected. Fix results retain their full
per-record outcomes and aggregate counts.

`rules()` exposes `id`, default `severity`, `fixable`, `fix_scope` (`record` or
`dataset`), `description`, and `fix_description`. Metadata comes from the rule
classes. A fixable rule may still drop records that cannot be repaired. The
context-window rule can emit warnings when counts are approximate and an info
note; rule metadata shows its default error severity. Eval overlap runs only
when an eval path is supplied. Schema cannot be disabled; unknown rule ids fail.

## MCP tools

Launch `parallelogram mcp`. It reads newline-delimited JSON-RPC 2.0 on stdin and
writes only JSON-RPC responses on stdout; diagnostics go to stderr. Relative
paths are relative to the server's working directory. Use absolute dataset paths
when the client's working directory is uncertain.

| Tool | Arguments | Result |
|---|---|---|
| `check_dataset` | Required `path`; optional `format`, `eval_path`, `disable` (array of ids), `limit` (nonnegative integer, default 50). | Check JSON with exact counts and bounded issues. |
| `fix_dataset` | Required `path`, `output`; optional `format`, `eval_path`, `disable`, `dry_run` (default false). | Fix JSON with summary, fixes by rule, fixer errors and outcomes. |
| `list_rules` | Empty object. | `{"rules": [...]}` containing the API rule list. |

Formats are `auto` (default), `openai-chat`, `sharegpt`, `prompt-completion`,
`alpaca`, and `preference`. MCP uses the default approximate context counter and
4096-token budget; use the Python API or CLI for a custom tokenizer/budget.

Successful tools return both `content: [{"type": "text", "text": "<compact JSON>"}]`
and `structuredContent` with the same object. Findings in a dataset are successful
tool results. Operational failures (including an in-place fix or forbidden
rule disabling) return `isError: true` and explanatory text. Invalid argument
shapes/types produce JSON-RPC -32602; unknown methods produce -32601; malformed
JSON produces -32700 and the server continues. EOF exits cleanly with code 0.

Initialization echoes supported protocol versions `2025-11-25`, `2025-06-18`,
`2025-03-26`, or `2024-11-05`, and otherwise selects `2025-11-25`. The server
advertises tools only; it supports ping and sends no notification responses.

## Client setup

Install Parallelogram in the environment your client launches. If the executable
is not on that client's PATH, replace `parallelogram` with its absolute path.
These examples configure the local stdio server; no API key is needed.

### Claude Code

```bash
claude mcp add parallelogram -- parallelogram mcp
```

See [Claude Code's MCP configuration](https://code.claude.com/docs/en/mcp).

### Codex

Run `codex mcp add parallelogram -- parallelogram mcp`, or add to `~/.codex/config.toml`:

```toml
[mcp_servers.parallelogram]
command = "parallelogram"
args = ["mcp"]
# Optional: cwd = "/absolute/path/to/dataset/project"
```

See [Codex MCP configuration](https://developers.openai.com/codex/mcp).

### Cursor

Add to your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "parallelogram": {
      "command": "parallelogram",
      "args": ["mcp"]
    }
  }
}
```

See [Cursor MCP configuration](https://cursor.com/docs/mcp).

## Paste into AGENTS.md or CLAUDE.md

```text
Run Parallelogram before fine-tuning and after every dataset edit.
Prefer: parallelogram check train.jsonl --json --limit 50.
Check exits: 0 no errors/warnings, 1 warnings, 2 errors/invalid arguments, 3 operational failure.
Fix exits: 0 all emitted, 1 partial fix, 2 none emitted; report exit 3 means baseline regression.
Inspect exact summary/by_rule counts and issues_truncated; use rules --json for repair guidance.
Repair with --fix --output clean.jsonl, never overwrite input, then recheck the output.
```
