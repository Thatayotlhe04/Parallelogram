# parallelogram

The linter for fine-tuning data. **If `parallelogram check` exits 0 with all rules enabled, your training run won't fail because of data.**

[![PyPI version](https://img.shields.io/pypi/v/parallelogram)](https://pypi.org/project/parallelogram/)
[![Python](https://img.shields.io/pypi/pyversions/parallelogram)](https://pypi.org/project/parallelogram/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](cli/LICENSE)

## Agents and Python (1.5)

```bash
parallelogram check train.jsonl --json --limit 50
parallelogram rules --json
parallelogram mcp
```

```python
from parallelogram import check, fix, rules

result = check("train.jsonl", limit=50)
payload = result.to_dict()
status = result.exit_code
fixed = fix("train.jsonl", output="clean.jsonl")
```

The API and CLI share their check/fix implementation. JSON retains
`schema_version: 1`, with exact `by_rule` counts and `issues_truncated`; limiting
issues never limits validation. Fixes require a separate output path and refuse
input/eval aliases. [Agent guide](cli/docs/agents.md): MCP tools, Claude Code/Codex/Cursor
configuration, Python errors, and a six-line AGENTS.md/CLAUDE.md snippet.

## The problem

Every fine-tuning framework assumes your data is clean. None of them verify it:

- **Axolotl** starts a run on malformed data and either crashes mid-way or — worse — completes silently while producing a broken model.
- **TRL** truncates samples that exceed the context window without telling you.
- **Unsloth** trains on duplicates that cause your model to memorize instead of generalize.

`parallelogram` sits between your raw dataset and your training run. It hard-blocks on anything that would silently corrupt training.

## Repository layout

```
.
├── cli/        — the Python CLI (the actual product)
└── landing/    — the parallelogram.dev marketing page
```

---

## Quickstart

Install from PyPI:

```bash
pip install parallelogram
parallelogram check data.jsonl
```

For an exact context-window token count (instead of approximate), install the tokenizer extras:

```bash
pip install 'parallelogram[tokenizer]'
```

This pulls in `tiktoken` (for OpenAI models) and HuggingFace `tokenizers` (for open-weight models). Without the extras the context-window check still runs — it uses an approximate length-based count and reports warnings instead of errors.

---

## Usage

### Basic check

```bash
parallelogram check data.jsonl
```

### Check with a model-specific tokenizer

Pass any OpenAI model name, HuggingFace repo, or short alias (`mistral`, `qwen`, `llama-3`, …):

```bash
parallelogram check data.jsonl \
  --tokenizer Qwen/Qwen2.5-7B \
  --max-seq-len 8192
```

### Write only clean records to a new file

```bash
parallelogram check data.jsonl --output clean.jsonl
```

### Check ShareGPT-style data directly

OpenAI/Qwen chat JSONL (`messages` with `role`/`content`) and ShareGPT-style
records (`conversations` with `from`/`value`) are auto-detected by default:

```bash
parallelogram check sharegpt.jsonl
```

### Check Alpaca and preference (DPO) data

Alpaca records (`instruction`, optional `input`, `output`, plus LLaMA-Factory's
`system` and `history`) and preference pairs (`prompt`, `chosen`, `rejected`,
or TRL's implicit-prompt shape where `chosen` and `rejected` are full
conversations) are auto-detected too. Every rule checks both sides of a
preference pair:

```bash
parallelogram check dpo.jsonl
parallelogram check alpaca.jsonl --fix --output clean.jsonl   # written back as Alpaca
```

### Catch train/eval leakage

```bash
parallelogram check train.jsonl --eval eval.jsonl
```

Flags train records whose whole content, or whose prompt, also appears in the
eval set. `--fix` drops them from the train output; the eval file is never touched.
Eval records are auto-detected independently of `--format`. Matching trims message
edges but preserves case and internal whitespace. An empty eval file is allowed;
invalid JSON, invalid schema, or a read failure aborts with exit 3 and a stderr
diagnostic, before any output dataset is written.

### Check prompt/completion data

`prompt-completion` is auto-detected alongside OpenAI/Qwen chat and ShareGPT,
including files that mix these formats. Both string and conversational pairs
are accepted; either half may independently be a string or message list:

```json
{"prompt": "What is 2 + 2?", "completion": "4."}
{"prompt": [{"role": "user", "content": "Capital of Botswana?"}], "completion": [{"role": "assistant", "content": "Gaborone."}]}
```

```bash
parallelogram check prompt_completion.jsonl
parallelogram check prompt_completion.jsonl --format prompt-completion
parallelogram check prompt_completion.jsonl --fix --output repaired.jsonl
```

Missing halves and values other than strings/lists are schema errors. Fixed
output keeps standard pairs as strings and conversational pairs as lists.
Conversational completion is the trailing run of assistant messages, with earlier
turns in prompt. A mixed record keeps a string half when it still contains one
matching turn; otherwise that half becomes a list to preserve every message. Extra top-level keys
may be omitted by fixes.

### Mechanical repair with `--fix`

`--fix` attempts free, local, network-free repair on everything it can touch:

```bash
parallelogram check data.jsonl --fix --output clean.jsonl
```

After fixes are applied, the dataset is **re-validated**. Anything still erroring is dropped. The CLI tells you exactly what happened:

```
✓ encoding · 4 fixes
✓ duplicates · 12 fixes

✗ dropped:
    data.jsonl:23 → roles (unfixable)
    data.jsonl:147 → schema (unfixable)

547 records  531 unchanged  4 fixed  11 dropped  1 unparseable
```

Fixes applied in order:

| Step | Fix |
|------|-----|
| 1 | **encoding** — strip BOM markers, replace mojibake (`donâ€™t` → `don't`) |
| 2 | **empty-content** — drop empty/whitespace-only message turns |
| 3 | **context-window** — truncate the longest user message until the record fits |
| 4 | **duplicates** — keep the first occurrence, drop subsequent |

Use `--dry-run` to preview without writing output:

```bash
parallelogram check data.jsonl --fix --dry-run
```

---

## Rules

| id | severity | catches |
|----|----------|---------|
| `schema` | error | malformed records, missing fields, wrong types |
| `roles` | error | bad role sequences — system out of place, no alternation, doesn't end on assistant |
| `empty-content` | error | empty or whitespace-only message content |
| `context-window` | error / warning | records exceeding `--max-seq-len` (TRL truncates these silently) — error with an exact tokenizer, warning when the count is approximate |
| `duplicates` | error | exact-content duplicate records (cause memorization → poor generalization) |
| `encoding` | warning | BOM markers, mojibake patterns |
| `secrets` | error | API keys, tokens and private keys (OpenAI, Anthropic, AWS, GitHub, Hugging Face, Slack, Google, Stripe). `--fix` redacts them; matches are never printed in full |
| `pii` | warning | email addresses, phone numbers, Luhn-valid card numbers. Example domains and 555 numbers are ignored; `--fix` replaces them with placeholders |
| `preference` | error | preference (DPO) pairs whose chosen and rejected answers are identical |
| `eval-overlap` | error | with `--eval`: train records whose content or prompt also appears in the eval set. `--fix` drops them |

### Disabling rules

Rules can be disabled by id. Three constraints apply:

- `schema` **cannot be disabled** — every other rule depends on its structural guarantees.
- Unknown ids are **rejected** — typos like `--disable encding` exit non-zero with a list of valid options rather than silently doing nothing.
- Any disabled rule triggers a **loud stderr warning** naming exactly which ones and reminding you the exit-0 guarantee no longer applies. The JSON report (`disabled_rules` field) also surfaces this so CI tooling can refuse to merge a PR that disabled rules.

```bash
parallelogram check data.jsonl --disable encoding --disable duplicates
```

---

## CLI options

| Flag | Description |
|------|-------------|
| `--version` | Show the CLI version and exit (use before a command). |
| `--format`, `-f` | Dataset format: `auto` (default), `openai-chat`, `sharegpt`, `prompt-completion`, `alpaca`, or `preference`. |
| `--eval` | Eval set to check for train/eval overlap (`check` and `report`). |
| `--tokenizer`, `-t` | Model or tokenizer for the context-window check — an OpenAI model (`gpt-4o`) or an HF repo/alias (`Qwen/Qwen2.5-7B`, `mistral`). Omit for an approximate count. |
| `--max-seq-len` | Token budget per record (default 4096). |
| `--output`, `-o` | Write error-free records to this file. With `--fix`, writes the repaired dataset. |
| `--fix` | Attempt mechanical repair of fixable issues. |
| `--dry-run` | With `--fix`, report what would change without writing. |
| `--json` | Machine-readable report on stdout. |
| `--limit` | Maximum check JSON issues to list; all counts remain exact. |
| `--disable` | Disable a rule by id. Repeatable. |
| `--no-color` | Plain output. |

---

## Exit codes

| Code | `check` | `check --fix` | `report` |
|------|---------|---------------|----------|
| `0`  | Clean — no errors or warnings. | All records emitted clean. | Clean. |
| `1`  | Warnings only. | Some records dropped (partial fix). | Warnings only. |
| `2`  | Errors. | Nothing fixable. | Errors. |
| `3`  | Operational failure, such as an unreadable or invalid `--eval` file. | Operational failure, such as an unreadable or invalid `--eval` file. | Invalid `--eval` input, or quality regressed vs `--baseline`. |

These map directly to CI gates without any extra wiring.

### Example CI step

```yaml
- name: Validate training data
  run: |
    pip install 'parallelogram[tokenizer]'
    parallelogram check data/train.jsonl \
      --tokenizer mistral \
      --max-seq-len 32768
```

---

## GitHub Action

Validate a dataset and add a Markdown health report to the workflow summary:

```yaml
name: Validate training data
on: [push, pull_request]
permissions:
  contents: read
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Thatayotlhe04/Parallelogram@v1
        with:
          path: data/train.jsonl
          format: auto
          max-seq-len: '4096'
          fail-on: error
          # tokenizer: Qwen/Qwen2.5-7B
          # baseline: baseline/report.json
          # eval: data/eval.jsonl
          # python-version: '3.12'
```

`@v1` tracks the 1.x releases. The action installs the CLI from its own checkout, so
its version matches the selected ref. Setting `tokenizer` installs the tokenizer
extras. The required `path` points to your checked-out dataset; all other inputs
are optional, with defaults shown above (tokenizer and baseline default to empty).
Errors (exit 2), invalid eval inputs and baseline regressions (exit 3) always fail. Warning-only reports
(exit 1) pass by default; use `fail-on: warning` to fail on them too. The report is
appended to the job summary even when validation fails.

## Stability

From 1.0 until 2.0, command names and flags, rule ids, exit codes 0/1/2/3,
and JSON fields at `schema_version: 1` are stable. Removing or renaming any of
these requires a major release. New rules and formats may be added in minor
releases. See the [JSON output contract](cli/docs/json-output.md) for every field and its meaning.
Reports saved by 0.4.x remain usable as baselines.

## Development

```bash
cd cli
pip install -e '.[dev]'

# Run the full test suite
pytest tests/

# Smoke-test the engine without any external dependencies
python smoke.py        # 32 stdlib-only checks (all six rules)
python smoke_fix.py    # 22 stdlib-only checks (fix tier)

# Lint
ruff check src/
```

---

## Landing page

Static HTML/CSS/JS for `parallelogram.dev`. No build step.

```bash
cd landing
python -m http.server 8000
# then open http://localhost:8000
```

Deploy by dropping `landing/` on Vercel, Netlify, Cloudflare Pages, or GitHub Pages and pointing `parallelogram.dev` at it.

---

## Status

**CLI v1.5.1, stable** — local pre-training validation with nine default rules and optional eval-overlap detection, mechanical `--fix`, model-specific tokenizers, chat/ShareGPT/prompt-completion/Alpaca/preference auto-detection, versioned JSON output, a Python API and an MCP server for coding agents, and a composite GitHub Action.

**Landing page** — [parallelogram.dev](https://parallelogram.dev): checks, the check/fix/gate workflow, changelog and FAQ, with privacy policy and terms of use. No cookies.

No telemetry. No backend. No upload boundary. Pure local.

---

## Roadmap

- ~~Prompt/completion format support~~ shipped in v1.0
- Opt-in anonymized error-type analytics — informs whether the SLM-fix tier is worth building
- `--fix --slm` paid hosted tier — repairs broken role sequences and incomplete assistant turns (gated on traction data from the analytics phase)
- Additional validation rules based on user feedback

---

## License

Apache 2.0 — see [`cli/LICENSE`](cli/LICENSE).
