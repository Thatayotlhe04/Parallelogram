#!/usr/bin/env bash
# Preserve report failures while allowing warning-only reports by default.
set -euo pipefail

case "$FAIL_ON" in
  error|warning) ;;
  *) printf '%s\n' 'fail-on must be error or warning' >&2; exit 2 ;;
esac

args=(report --markdown --format "$DATASET_FORMAT" --max-seq-len "$MAX_SEQ_LEN")
if [[ -n "$TOKENIZER" ]]; then
  args+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "$BASELINE" ]]; then
  args+=(--baseline "$BASELINE")
fi
if [[ -n "${EVAL_PATH:-}" ]]; then
  args+=(--eval "$EVAL_PATH")
fi
# A filename beginning with '-' must remain a positional argument.
args+=(-- "$DATASET_PATH")

code=0
out="$(mktemp)"
parallelogram "${args[@]}" > "$out" || code=$?
cat "$out" >> "$GITHUB_STEP_SUMMARY"
if [[ "$code" -eq 1 && "$FAIL_ON" == error ]]; then
  # Exit 1 means warnings only, and only when a report was written. Anything
  # else that exits 1 (a crash in an older CLI, a broken install) must fail.
  if grep -q '^## parallelogram report' "$out"; then
    exit 0
  fi
  printf '%s\n' 'parallelogram exited 1 without writing a report; treating it as a failure' >&2
  exit 3
fi
# Includes 2/3 and unexpected failures such as a missing executable.
exit "$code"
