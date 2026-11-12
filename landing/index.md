---
title: Parallelogram
description: Machine-readable overview of parallelogram.dev, for AI agents and for anyone who wants the landing page as text.
canonical: https://parallelogram.dev/
markdown_url: https://parallelogram.dev/index.md
llms_url: https://parallelogram.dev/llms.txt
agent_guide: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/agents.md
version: 1.6.1
updated: 2026-11-08
integrations: https://parallelogram.dev/integrations.html
---

# Parallelogram

> The linter for LLM fine-tuning data. A local CLI, Python API and MCP server that validates and mechanically repairs fine-tuning JSONL before training. If `parallelogram check` exits 0 with all rules on, the training run won't fail because of the data. Free and open source (Apache-2.0).

It runs on the user's machine: no telemetry, no network calls, no uploads. Python 3.10+, runtime dependencies typer and rich.

```bash
pip install parallelogram
```

## Use it as an MCP server

```bash
parallelogram integrate claude-code    # writes .mcp.json; also cursor, vscode, gemini
parallelogram integrate codex          # prints setup; also windsurf, cline, zed
parallelogram integrate --list
codex mcp add parallelogram -- parallelogram mcp
```

`integrate` writes project-scoped config only, merges into existing JSON, never touches user-level config, and refuses invalid JSON (exit 2). Add `--agents-md` to put the run instructions in AGENTS.md (CLAUDE.md for Claude Code), `--dry-run` to preview. Any other MCP client: a stdio server with command `parallelogram` and args `["mcp"]`.

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
parallelogram check a.jsonl b.jsonl --json       # several files: a JSON array, highest exit code
```

pre-commit: `repo: https://github.com/Thatayotlhe04/Parallelogram`, `rev: cli-v1.6.0`, hook id `parallelogram`. GitHub Actions: `uses: Thatayotlhe04/Parallelogram@v1` with `path:`.

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

Tool calling, in both shapes trainers use: OpenAI's (`tools`, `tool_calls` with ids and JSON-string arguments, `tool` replies with `tool_call_id`, legacy `functions`/`function_call`) and Hugging Face's for TRL, Axolotl and Unsloth (id-less calls, object arguments, replies answered in order, `tools` as a list or JSON string). ShareGPT `function_call`/`observation` turns as LLaMA-Factory writes them. A conversation may end on the assistant's call.

## Checks

| Rule | Severity | --fix |
| --- | --- | --- |
| schema | error | Malformed records are left out |
| roles | error | Records with an invalid role order, or tool calls and replies that don't pair up, are dropped |
| tool-calls | error | Records with malformed tool definitions or calls, undeclared functions, or non-object arguments are dropped; argument values are never printed |
| empty-content | error | Empty turns are removed |
| context-window | error | The longest user turns are truncated to fit |
| duplicates | error | The first copy is kept |
| encoding | warning | BOMs and mojibake are repaired |
| preference | error | Pairs with identical chosen and rejected answers are dropped |
| secrets | error | API keys, tokens and private keys are redacted, in every field of a record |
| pii | warning | Emails, phone numbers and card numbers are replaced with placeholders: in training text, and in fields named for them (`email`, `phone`, `card`) |
| eval-overlap | error | Train records that repeat the eval set are dropped (only with `--eval`) |

Without `--tokenizer`, token counts are estimates and an estimated overflow is a warning, not an error.

## For agents

- Run `check` before fine-tuning and after every dataset edit.
- Judge the result by `exit_code`, not `summary.clean`, which is also false when the only finding is an informational note.
- Fix into a separate file, then check that file.
- Never overwrite the input or the eval set.

## Product walkthrough

The CLI panel automatically demonstrates a small customer-support dataset in
an editor alongside a terminal. Three examples contain one duplicate and one
synthetic email address. The check highlights each issue in its source record;
the fix writes two records to a separate clean.jsonl file, replacing the email
with [REDACTED_EMAIL]. The original stays untouched. Re-checking the new file
returns zero errors and zero warnings.

The input, output file and terminal stdout are captured from Parallelogram
1.6.1. Plain-language captions connect each command to the visible file change.
The demonstration loops automatically without buttons or timers. It pauses
when offscreen, in a hidden tab, or focused for keyboard reading. Reduced
motion shows the completed result and full transcript. Without JavaScript,
the explanation and transcript remain available.

## Model comparisons

The landing page leads with the impact of training-data defects: empty answers
reduced accuracy by 22.4 percentage points in the 30% injection test (4.6 at
10%). This is damage measured in a controlled model, not accuracy recovered by
Parallelogram. The practical benefit is flagging these checkable defects before
training, separating eval data, redacting secrets and re-checking repairs.

Additional views compare API costs and self-hosted model ownership. It is not
an intelligence ranking. Pricing is a fixed snapshot from 12 November 2026.
One task means 1,500 input and 200 billed output tokens, including reasoning.

- Flagship APIs: GPT-6 Astra ($10/$50 per million input/output tokens), Claude
  Fable 5.2 ($8/$40), Gemini 4 Carbon ($2/$10, promotional through Nov 30).
- Efficient APIs: GPT-6 Luna ($0.10/$0.50), Claude Haiku 5.5 ($0.10/$0.50),
  Gemini 3.1 Flash-Lite ($0.25/$1.50).
- The self-hosted fine-tune is an editable $6,000/month scenario, including
  hosting, maintenance and amortized training. It assumes sufficient capacity
  for the selected workload and makes no claim of matched model quality.
- A separate controlled 383K-parameter model test shows 100% accuracy on clean
  data, 77.6% with 30% empty answers, and 68.3% with systematic wrong labels.
  Five seeds, greedy decoding, 600 held-out extraction examples. It does not
  measure accuracy recovered by Parallelogram or production-model performance.
- Parallelogram 1.6.1 detected 98–100% of injected rows in 18 of 23 defect types
  in a separate detection test, including warnings. Semantic defects remain.

Official pricing sources:
https://developers.openai.com/api/docs/pricing
https://platform.claude.com/docs/en/about-claude/pricing
https://aistudio.google.com/docs/pricing

## Links

- Source: https://github.com/Thatayotlhe04/Parallelogram
- PyPI: https://pypi.org/project/parallelogram/
- Agent guide: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/agents.md
- JSON contract: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/docs/json-output.md
- Changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
