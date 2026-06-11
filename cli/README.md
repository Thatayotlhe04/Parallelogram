# parallelogram

> Strict validator for fine-tuning datasets. Run it before you train.

Every fine-tuning framework assumes your data is clean. None of them verify it.
Axolotl will start a run on malformed data and either crash mid-way or — worse —
complete silently while producing a broken model. TRL will truncate samples that
exceed the context window without telling you. Unsloth will train on duplicates
that cause your model to memorize instead of generalize.

`parallelogram` sits between your raw dataset and your training run. It hard-blocks
on anything that would silently corrupt training. **If it exits 0 with all rules enabled,
your run won't fail because of data.**

## Install

```bash
pip install parallelogram
```

For an **exact** context-window count, install the tokenizer extras — HuggingFace
`tokenizers` for open-weight models and `tiktoken` for OpenAI models:

```bash
pip install 'parallelogram[tokenizer]'
```

Without the extras (or for a model with no offline tokenizer, like Claude), the
context-window check still runs using an approximate length-based count.

## Use

```bash
parallelogram check data.jsonl
```

With a model-specific tokenizer for an exact context-window count — an OpenAI model,
or any HuggingFace repo or short alias (`mistral`, `qwen`, `llama-3`, …):

```bash
parallelogram check data.jsonl \
  --tokenizer Qwen/Qwen2.5-7B \
  --max-seq-len 8192
```

Omit `--tokenizer` and the check still runs with an approximate count, reported as
warnings instead of errors.

Write only the clean records to a new file:

```bash
parallelogram check data.jsonl --output clean.jsonl
```

## `--fix` — mechanical repair

When `parallelogram check` finds errors, `--fix` attempts to repair what it can without
touching the model. Mechanical fixes are free, local, and require no network call.

```bash
parallelogram check data.jsonl --fix --output clean.jsonl
```

Fixes applied in order:

1. **encoding** — strip BOM markers, replace mojibake (`donâ€™t` → `don't`)
2. **empty-content** — drop empty/whitespace-only message turns
3. **context-window** — truncate the longest user message until the record fits
4. **duplicates** — keep the first occurrence, drop subsequent

After fixes are applied, the dataset is **re-validated**. Anything still erroring is
dropped from the output. The CLI tells you exactly what was unchanged, fixed, and
dropped:

```
✓ encoding · 4 fixes
✓ duplicates · 12 fixes

✗ dropped:
    data.jsonl:23 → roles (unfixable)
    data.jsonl:147 → schema (unfixable)

547 records  531 unchanged  4 fixed  11 dropped  1 unparseable
```

Errors that need understanding the content (broken role sequences, incomplete
assistant turns) are not fixable mechanically. These will be addressed by a hosted
SLM tier in a future release. For now, they are dropped.

Use `--dry-run` to preview without writing:

```bash
parallelogram check data.jsonl --fix --dry-run
```

## `--disable` and the exit-0 guarantee

Rules can be disabled by id (e.g. `--disable encoding`), but with three constraints:

- The `schema` rule cannot be disabled. Every other rule depends on its structural
  guarantees, and disabling it would let other rules silently no-op on malformed records.
- Unknown rule ids are rejected. Typos like `--disable encding` exit non-zero with a
  list of valid options rather than silently doing nothing.
- Whenever any rule is disabled, a loud stderr warning names exactly which ones, and
  reminds you that the exit-0 guarantee no longer applies. The terminal output and
  JSON report (`disabled_rules` field) both surface this so CI tooling can refuse to
  merge a PR that disabled rules.

The guarantee is precise: **a clean exit with all rules enabled** means your run won't
fail because of data. A clean exit with rules disabled means only that the rules you
left enabled passed — which may or may not be enough.

## Options

| Flag | Description |
|------|-------------|
| `--format`, `-f` | Dataset format. Only `openai-chat` in v0.1. |
| `--tokenizer`, `-t` | Model or tokenizer for the context-window check — an OpenAI model (`gpt-4o`), or an HF repo/alias (`Qwen/Qwen2.5-7B`, `mistral`). Optional: omit for an approximate count. |
| `--max-seq-len` | Token budget per record (default 4096). |
| `--output`, `-o` | Write error-free records to this file. With `--fix`, writes the repaired dataset. |
| `--fix` | Attempt mechanical repair of fixable issues. |
| `--dry-run` | With `--fix`, report what would change without writing. |
| `--json` | Machine-readable report on stdout. |
| `--disable` | Disable a rule by id. Repeatable. |
| `--no-color` | Plain output. |

## Exit codes

| Code | Meaning (check) | Meaning (--fix) |
|------|---------|---------|
| `0`  | Clean. | All records emitted clean. |
| `1`  | Warnings only. | Some records dropped (partial fix). |
| `2`  | Errors. | Nothing fixable. |

These map directly to CI gates without any extra wiring.

## Rules

| id | severity | catches |
|----|----------|---------|
| `schema` | error | malformed records, missing fields, wrong types |
| `roles` | error | bad role sequences (system out of place, no alternation, doesn't end on assistant) |
| `empty-content` | error | empty or whitespace-only message content |
| `context-window` | error / warning | records exceeding `max_seq_len` (TRL truncates these silently) — error with an exact tokenizer, warning when the count is approximate |
| `duplicates` | error | exact-content duplicate records (memorization → poor generalization) |
| `encoding` | warning | BOM markers, mojibake patterns |

## Status

v0.3 — solo dev, local, pre-training run. No telemetry, no network, no upload boundary.

## Roadmap

- ShareGPT and raw-completion formats
- ~~`--fix` mechanical tier (dedupe, truncate, normalize encoding)~~ ✓ shipped in v0.2
- ~~Model-specific tokenizers (tiktoken/HF) with approximate fallback~~ ✓ shipped in v0.3
- Opt-in anonymized error-type analytics (informs SLM tier scope)
- `--fix --slm` paid hosted tier — repairs broken role sequences, incomplete turns

## License

Apache-2.0
