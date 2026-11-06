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
# A filename beginning with '-' must remain a positional argument.
args+=(-- "$DATASET_PATH")

code=0
parallelogram "${args[@]}" >> "$GITHUB_STEP_SUMMARY" || code=$?
if [[ "$code" -eq 1 && "$FAIL_ON" == error ]]; then
  exit 0
fi
# Includes 2/3 and unexpected failures such as a missing executable.
exit "$code"
