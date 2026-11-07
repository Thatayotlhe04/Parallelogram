---
title: Parallelogram
description: Machine-readable overview of parallelogram.dev, for AI agents and for anyone who wants the landing page as text.
canonical: https://parallelogram.dev/
markdown_url: https://parallelogram.dev/index.md
llms_url: https://parallelogram.dev/llms.txt
agent_guide: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/agents.md
version: 1.5.1
updated: 2026-11-07
---

# Parallelogram

> The linter for LLM fine-tuning data. A local CLI, Python API and MCP server that validates and mechanically repairs fine-tuning JSONL before training. If `parallelogram check` exits 0 with all rules on, the training run won't fail because of the data. Free and open source (Apache-2.0).

It runs on the user's machine: no telemetry, no network calls, no uploads. Python 3.10+, runtime dependencies typer and rich.

```bash
pip install parallelogram
```

## Use it as an MCP server

```bash
claude mcp add parallelogram -- parallelogram mcp    # Claude Code
codex mcp add parallelogram -- parallelogram mcp     # Codex
```

Cursor, Windsurf, Cline and other MCP clients: a stdio server with command `parallelogram` and args `["mcp"]`.

- `check_dataset` {path, format?, eval_path?, disable?, limit?}: validate without changing files. Counts are exact; at most `limit` issues are listed (default 50), errors first. Returns `exit_code`.
- `fix_dataset` {path, output, format?, eval_path?, disable?, dry_run?, overwrite?, limit?}: write a repaired copy to `output`, which must be inside the server's root (its working directory, or `--root`; relative paths resolve against it) and must not already exist unless `overwrite` is true. Never writes over the input or eval file. Lists only dropped, unparseable and fixed records. Returns `exit_code`.
- `list_rules` {}: every rule with its severity, whether it is fixable, and what the fix does.

## Use it from the shell

```bash
parallelogram check train.jsonl --json --limit 50
parallelogram check train.jsonl --eval eval.jsonl --json --limit 50
parallelogram check train.jsonl --fix --output clean.jsonl
parallelogram rules --json
parallelogram report train.jsonl --baseline last.json
```

`check` exits 0 with no errors or warnings, 1 with warnings only, 2 with errors, 3 on an operational failure such as an invalid `--eval` file. With `--fix`: 0 every record emitted, 1 some dropped or unparseable, 2 nothing emitted. `report --baseline` exits 3 when quality regressed.

## Use it from Python

```python
from parallelogram import check, fix

result = check("train.jsonl", eval_path="eval.jsonl", limit=50)
result.exit_code, result.to_dict()   # to_dict() equals the CLI's --json output
fix("train.jsonl", output="clean.jsonl")
```

## Formats

Auto-detected per line: OpenAI chat (`messages`), ShareGPT (`conversations`), prompt/completion, Alpaca (`instruction`, `input`, `output`, plus `system` and `history`) and preference/DPO pairs (`prompt`, `chosen`, `rejected`). `--fix --output` writes every record back in its own format.

## Checks

| Rule | Severity | --fix |
| --- | --- | --- |
| schema | error | Malformed records are left out |
| roles | error | Records with an invalid role order are dropped |
| empty-content | error | Empty turns are removed |
| context-window | error | The longest user turns are truncated to fit |
| duplicates | error | The first copy is kept |
| encoding | warning | BOMs and mojibake are repaired |
| preference | error | Pairs with identical chosen and rejected answers are dropped |
| secrets | error | API keys, tokens and private keys are redacted, in every field of a record |
| pii | warning | Emails, phone numbers and card numbers in training text are replaced with placeholders |
| eval-overlap | error | Train records that repeat the eval set are dropped (only with `--eval`) |

Without `--tokenizer`, token counts are estimates and an estimated overflow is a warning, not an error.

## For agents

- Run `check` before fine-tuning and after every dataset edit.
- Judge the result by `exit_code`, not `summary.clean`, which is also false when the only finding is an informational note.
- Fix into a separate file, then check that file.
- Never overwrite the input or the eval set.

## Links

- Source: https://github.com/Thatayotlhe04/Parallelogram
- PyPI: https://pypi.org/project/parallelogram/
- Agent guide: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/agents.md
- JSON contract: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/json-output.md
- Changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
