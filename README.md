# parallelogram

The linter for fine-tuning data. **If `parallelogram check` exits 0 with all rules enabled, your training run won't fail because of data.**

[![PyPI version](https://img.shields.io/pypi/v/parallelogram)](https://pypi.org/project/parallelogram/)
[![Python](https://img.shields.io/pypi/pyversions/parallelogram)](https://pypi.org/project/parallelogram/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](cli/LICENSE)

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
| `--format`, `-f` | Dataset format: `auto` (default), `openai-chat`, or `sharegpt`. |
| `--tokenizer`, `-t` | Model or tokenizer for the context-window check — an OpenAI model (`gpt-4o`) or an HF repo/alias (`Qwen/Qwen2.5-7B`, `mistral`). Omit for an approximate count. |
| `--max-seq-len` | Token budget per record (default 4096). |
| `--output`, `-o` | Write error-free records to this file. With `--fix`, writes the repaired dataset. |
| `--fix` | Attempt mechanical repair of fixable issues. |
| `--dry-run` | With `--fix`, report what would change without writing. |
| `--json` | Machine-readable report on stdout. |
| `--disable` | Disable a rule by id. Repeatable. |
| `--no-color` | Plain output. |

---

## Exit codes

| Code | `check` | `check --fix` |
|------|---------|---------------|
| `0`  | Clean — no errors or warnings. | All records emitted clean. |
| `1`  | Warnings only. | Some records dropped (partial fix). |
| `2`  | Errors. | Nothing fixable. |

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

**CLI v0.4.2** — local pre-training validation with six rules, mechanical `--fix`, model-specific tokenizers, ShareGPT-style auto-detection, and the `report` CI regression gate. Published to PyPI. All smoke and end-to-end suites passing.

**Landing page v0.4** — quickstart shows `--fix`, format support, one-click pip install copy, cookie consent banner, privacy policy, and terms of use.

No telemetry. No backend. No upload boundary. Pure local.

---

## Roadmap

- raw-completion format support
- Opt-in anonymized error-type analytics — informs whether the SLM-fix tier is worth building
- `--fix --slm` paid hosted tier — repairs broken role sequences and incomplete assistant turns (gated on traction data from the analytics phase)
- Additional validation rules based on user feedback

---

## License

Apache 2.0 — see [`cli/LICENSE`](cli/LICENSE).
