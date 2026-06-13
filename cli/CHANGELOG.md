# Changelog

All notable changes to the `parallelogram` CLI. Versions correspond to
[PyPI releases](https://pypi.org/project/parallelogram/) and `cli-v*` git tags.

## 0.4.2 — 2026-06-13

### Changed
- `check` and `report` now default to `--format auto`, accepting both
  OpenAI/Qwen chat JSONL (`messages` with `role`/`content`) and
  ShareGPT-style records
  (`{"conversations": [{"from": ..., "value": ...}]}`) directly.
- Auto-detected ShareGPT-style records preserve their source shape when
  `--fix --output` writes repaired data, so output stays in `conversations`
  form instead of being re-serialized as OpenAI chat.
- `report` includes detected source-format counts in its format breakdown.

## 0.4.1 — 2026-06-12

### Added
- `parallelogram report <path>` — aggregate dataset health for humans and CI,
  built from one validation pass plus a dry fix pass so it can never disagree
  with `check` / `check --fix`:
  - clean/error/warning/unparseable record counts and issues by rule, with
    fixable counts and a `--fix` projection (emit/fix/drop)
  - token risk vs `--max-seq-len`: over-budget and at-risk (≥85%) records,
    median and max, labeled exact or estimated by the same counter `check` uses
  - duplicate clusters (count, duplicate records, largest cluster) and a format
    breakdown (roles, turns per record, ends-on-assistant)
  - three outputs: terminal, `--json`, and `--markdown` (drops straight into
    `$GITHUB_STEP_SUMMARY`); `--out` writes the report to a file
- CI regression gate: `report --baseline previous.json` exits 3 and names
  exactly what got worse. Comparison is rate-based (errors per record, dupes
  per record, over-budget per record, dropped per record, clean fraction) so
  dataset growth alone never reads as a regression.
- Exit codes documented as stable (0/1/2/3) and GitHub Actions examples in the
  README.

### Fixed
- A perfectly clean dataset now exits 0 on a default install. The one-time
  "context-window counts are approximate" advisory was a warning, which broke
  the exit-0 guarantee and CI gates unless the tokenizer extras were installed;
  it is now informational (visible, never affects the exit code).
- Report tests no longer depend on the installed `click` version
  (`CliRunner(mix_stderr=False)` was removed in click ≥ 8.2).

## 0.4.0 — 2026-06-11

### Added
- ShareGPT format support: `--format sharegpt` alongside the default
  `openai-chat`. ShareGPT records (`{"conversations": [{"from", "value"}]}`)
  are normalized to the internal OpenAI-chat shape at the parse boundary
  (top-level `system` lifted, `human` → `user`, `gpt` → `assistant`; unknown
  `from` values pass through so the schema rule flags them), so every rule,
  the fixer, and the report run unchanged.
- `--fix --output` writes ShareGPT-shaped records back out when the input was
  ShareGPT, round-tripping the format.

## 0.3.0 — 2026-06-11

### Added
- Model-specific token counting for the context-window rule: tiktoken for
  OpenAI models, HuggingFace tokenizers for open-weight models (exact repo or
  short alias), and a length-based approximate fallback when no tokenizer is
  available (e.g. Claude). The check always runs instead of silently disabling
  itself.
- `tiktoken` added to the `[tokenizer]` extra.

### Changed
- Context-window overflows are errors when counted exactly and warnings when
  estimated, with a one-time note explaining the estimate.

## 0.2.1 — 2026-06-11

### Fixed
- Fixer exceptions are no longer silently swallowed: a crashing fix rule is
  recorded in `FixReport.fixer_errors`, printed to stderr, and exits 1 instead
  of hiding behind a clean exit.
- `check --output` no longer labels warning-carrying records (mojibake/BOM) as
  "clean": they are copied verbatim, the output says so and points at `--fix`.
- Renamed the `format` parameter to `dataset_format` (it shadowed the builtin)
  and dropped stale "v0.1" strings from help and error text.

## 0.2.0 — 2026-06-11

First public release on PyPI.

- `parallelogram check <dataset.jsonl>` — validate OpenAI chat-format
  fine-tuning data locally: schema, empty/whitespace content, role order,
  duplicates, encoding issues (mojibake/BOM), context-window overflow.
- `check --fix` — apply safe automatic fixes and write a cleaned copy.
- Runs entirely offline; data never leaves the machine.
