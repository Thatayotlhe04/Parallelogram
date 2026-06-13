#!/usr/bin/env bash
# Publish GitHub releases with notes for every existing cli-v* tag.
# Requires: gh auth login (or GH_TOKEN). Safe to re-run; skips existing releases.
set -euo pipefail

repo="Thatayotlhe04/Parallelogram"

release() {
  local tag="$1" title="$2"
  if gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
    echo "skip: $tag already has a release"
    return
  fi
  gh release create "$tag" --repo "$repo" --title "$title" --notes-file -
  echo "created: $tag"
}

release cli-v0.2.0 "parallelogram 0.2.0 — first PyPI release" <<'EOF'
First public release on PyPI.

- `parallelogram check <dataset.jsonl>` — validate OpenAI chat-format fine-tuning data locally: schema, empty/whitespace content, role order, duplicates, encoding issues (mojibake/BOM), context-window overflow.
- `check --fix` — apply safe automatic fixes and write a cleaned copy.
- Runs entirely offline; data never leaves the machine.

Install: `pip install parallelogram`
EOF

release cli-v0.2.1 "parallelogram 0.2.1 — surface fixer crashes, honest --output" <<'EOF'
Patch release.

**Fixed**
- Fixer exceptions are no longer silently swallowed: a crashing fix rule is recorded in `FixReport.fixer_errors`, printed to stderr, and exits 1 instead of hiding behind a clean exit.
- `check --output` no longer labels warning-carrying records (mojibake/BOM) as "clean": they are copied verbatim, the output says so and points at `--fix`.
- Renamed the `format` parameter to `dataset_format` (it shadowed the builtin) and dropped stale "v0.1" strings from help and error text.

Full changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
EOF

release cli-v0.3.0 "parallelogram 0.3.0 — model-specific tokenizers" <<'EOF'
**Added**
- Model-specific token counting for the context-window rule: tiktoken for OpenAI models, HuggingFace tokenizers for open-weight models (exact repo or short alias), and a length-based approximate fallback when no tokenizer is available (e.g. Claude). The check always runs instead of silently disabling itself.
- `tiktoken` added to the `[tokenizer]` extra.

**Changed**
- Context-window overflows are errors when counted exactly and warnings when estimated, with a one-time note explaining the estimate.

Full changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
EOF

release cli-v0.4.0 "parallelogram 0.4.0 — ShareGPT format support" <<'EOF'
**Added**
- ShareGPT format support: `--format sharegpt` alongside the default `openai-chat`. ShareGPT records (`{"conversations": [{"from", "value"}]}`) are normalized to the internal OpenAI-chat shape at the parse boundary (top-level `system` lifted, `human` → `user`, `gpt` → `assistant`; unknown `from` values pass through so the schema rule flags them), so every rule, the fixer, and the report run unchanged.
- `--fix --output` writes ShareGPT-shaped records back out when the input was ShareGPT, round-tripping the format.

Full changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
EOF

release cli-v0.4.1 "parallelogram 0.4.1 — report command + CI regression gate" <<'EOF'
**Added**
- `parallelogram report <path>` — aggregate dataset health for humans and CI, built from one validation pass plus a dry fix pass so it can never disagree with `check` / `check --fix`: record counts, issues by rule with fixable counts and a `--fix` projection, token risk vs `--max-seq-len` (over-budget and ≥85% at-risk, exact or estimated), duplicate clusters, and a format breakdown. Three outputs: terminal, `--json`, `--markdown` (drops straight into `$GITHUB_STEP_SUMMARY`); `--out` writes to a file.
- CI regression gate: `report --baseline previous.json` exits 3 and names exactly what got worse. Comparison is rate-based, so dataset growth alone never reads as a regression.
- Exit codes documented as stable (0/1/2/3); GitHub Actions examples in the README.

**Fixed**
- A perfectly clean dataset now exits 0 on a default install: the one-time "context-window counts are approximate" advisory is informational instead of a warning, so it no longer breaks the exit-0 guarantee and CI gates without the tokenizer extras.
- Report tests no longer depend on the installed `click` version.

Full changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
EOF

release cli-v0.4.2 "parallelogram 0.4.2 — direct ShareGPT-style input" <<'EOF'
**Changed**
- `parallelogram check` and `parallelogram report` now default to `--format auto`, so OpenAI/Qwen chat JSONL (`messages` with `role`/`content`) and ShareGPT-style `conversations` records are accepted directly.
- Auto-detected ShareGPT-style records keep their source shape when `--fix --output` writes repaired data, preserving `conversations` instead of re-serializing as OpenAI chat.
- `report` now includes detected source-format counts in the format breakdown.

Full changelog: https://github.com/Thatayotlhe04/Parallelogram/blob/main/cli/CHANGELOG.md
EOF

echo "done"
