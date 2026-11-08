# Changelog

All notable changes to the `parallelogram` CLI. Versions correspond to
[PyPI releases](https://pypi.org/project/parallelogram/) and `cli-v*` git tags.

## 1.6.1 — 2026-11-08

Fixes from running 1.6 over popular public datasets:

- `null` tool fields (`tool_calls`, `function_call`, `tools`, `functions`,
  `tool_call_id`, a call's `id`) read as absent. Hugging Face `datasets` writes
  null for any column a row doesn't use, so ordinary chat rows exported from a
  mixed dataset failed `tool-calls` with exit 2.
- An empty `tool_calls: []` is a warning, not an error: chat templates treat it
  as no calls, OpenAI's API wants null or at least one. `--fix` no longer drops
  a record whose only unfixable issue is a warning.
- HuggingFaceH4/ultrafeedback_binarized-style rows (`prompt` plus full
  `chosen`/`rejected` conversations and a `messages` column) are read as
  preference pairs. The repeated prompt was being prepended a second time,
  failing `roles`. Every column round-trips in its original order.
- A string `prompt` beside full `chosen`/`rejected` conversations is read as
  metadata whichever turn it copies, as TRL's `maybe_extract_prompt` does.
  mlabonne/orpo-dpo-mix-40k keeps the opening turn on multi-turn rows (4,487
  false `roles` errors) and argilla/distilabel-capybara-dpo-7k-binarized keeps
  a chat-templated string.
- Preference pairs with a `conversation` column that holds metadata rather
  than turns (distilabel-capybara-dpo's `{"input", "output"}` pairs) are read
  as preference, not ShareGPT. Every row was failing `schema`.
- Hermes-format tool calls written in the text
  (`<tool_call>{...}</tool_call>`, NousResearch/hermes-function-calling-v1)
  are matched by the `tool` turns after them; 1,100 of 1,893 rows were
  failing `roles`. The 173 that end on tool replies the assistant never
  requested still fail.
- Issues in a preference row's own `messages` column name it `messages`, not
  an internal key.

Also:

- Listed in the official MCP Registry as `io.github.Thatayotlhe04/parallelogram`
  (`server.json`, and the ownership marker in the PyPI README).
- The GitHub Action's `@v1` tag now follows every 1.x release; it had stayed on
  1.0.0. Pin `@cli-v1.x.y` if you want upgrades to be explicit.
- More PyPI keywords; integration guide examples point at released tags.

## 1.6.0 — 2026-11-08

### Added
- Tool-calling datasets in both shapes trainers use: OpenAI's (call ids,
  JSON-string arguments, `tool_call_id` replies) and Hugging Face's, as in TRL,
  Axolotl and Unsloth (id-less calls, object arguments, replies answered in
  order, `tools` as a list or a JSON string). Parallel call/reply validation,
  optional tool-only assistant content, and legacy function-call support.
- A conversation may end on the assistant's tool call, as OpenAI's own
  function-calling fine-tuning examples do: the call is the training target.
- LLaMA-Factory ShareGPT `function_call`/`observation` normalization and inverse
  serialization, including JSON-string tool definitions.
- `tool-calls`: error diagnostics for malformed definitions/calls, arguments
  that are not an object, undeclared functions, and duplicate ids, without
  argument values.
- `parallelogram integrate <client>` for Claude Code, Cursor, VS Code and Gemini
  project MCP config; manual setup for Codex, Windsurf, Cline and Zed. Includes
  JSON merging, idempotence, `--agents-md`, `--dry-run`, and `--list`.
- Repository pre-commit hook `parallelogram` and root installation metadata.
- Integration documentation covering clients, pre-commit, the GitHub Action,
  and TRL, Axolotl, Unsloth, LLaMA-Factory and OpenAI training preflights.

### Changed
- `check` accepts several inputs, checks each, and returns the highest exit code.
  JSON is an array only for multiple inputs; `--output` requires one input.
- Context counts include tool definitions and calls. Duplicate hashing includes
  tool-call content, reply links and definitions.
- A clean no-op `--fix --output` preserves the complete source bytes, including
  whitespace, blank lines, CRLF, and an absent final newline. Unchanged records
  within repaired files also retain their original serialized content.
- Empty-turn repairs never remove turns from tool exchanges; broken records
  are dropped whole. Runtime dependencies and JSON schema version remain unchanged.

## 1.5.1 — 2026-11-07

Fixes from an independent leak audit of 1.5.0. Every fix has a regression
test. Upgrade if you run Parallelogram in CI or let an agent call it.

### Security
- `secrets` scans every string in a record, dict and message keys included,
  not only message text: ids, metadata and other extra fields, message names and
  tool-call arguments, in every format. Before, a key in a record's `id` passed
  `check` and `check --output` copied it out unchanged. `pii` checks the text a
  model trains on, now including the text parts of multimodal content; ids and
  metadata are left alone, because order numbers, SKUs and git remotes look
  like phone numbers and emails.
- One line with an integer over 4,300 digits, or nested thousands of levels
  deep, crashed `report` with exit 1, and the GitHub Action read exit 1 as
  "warnings only" and passed. Such lines are now invalid JSON lines like any
  other. Any unexpected crash exits 3, and the Action fails when the CLI exits
  1 without writing a report.
- `--fix` applies one kind of repair per pass in a fixed order: encoding,
  then secrets and PII, then empty turns and truncation. Truncating before
  redacting could leave most of a key in the output, too short to match again;
  repairing encoding first joins a key split by a stray BOM before it's
  redacted.
- An email address or phone number used as a role value is no longer printed
  in diagnostics, `report` or the Action summary. Email masks no longer show the
  domain (`j***@***.com`).
- MCP `fix_dataset` writes only inside the server's root (its working
  directory, or `parallelogram mcp --root DIR`), refuses symlinked outputs, and
  replaces an existing file only with `overwrite: true`. Relative paths resolve
  against that root. `path` and `eval_path` must be regular files outside
  `/proc`, `/sys` and `/dev`, so a pipe or `/dev/stdin` can't block the
  server, and request lines over 1,000,000 characters are rejected.
- Values in the Markdown summary (paths, roles, formats, tokenizer names) are
  fenced so they can't inject links or images into a job summary.
- New secret kinds: PGP private key blocks, AWS secret access keys next to
  their name (including the `"SecretAccessKey": "..."` JSON form), Slack
  webhooks and `xoxe-` tokens, Google OAuth client secrets, Hugging Face
  `api_org_` tokens, and keys that directly follow an underscore.

### Changed
- `--fix --output` keeps extra fields on ShareGPT and prompt/completion
  records (such as `id`, and per-turn fields like `weight`) instead of
  dropping them, as it already did for chat, Alpaca and preference records.
  When two redacted keys in one object would collide, the second becomes
  `[REDACTED_SECRET]_2` rather than overwriting the first.
- The publish workflow runs tests and the build without PyPI credentials;
  only a separate publish job can mint a token, and its actions are pinned to
  commit SHAs. The GitHub Action pins `actions/setup-python` to a commit SHA,
  since it runs inside your workflows.

## 1.5.0 — 2026-11-07

Parallelogram for coding agents. Claude Code, Codex, Cursor or your own
pipeline can run the checks, read a result sized for a context window, and fix
the data without anyone translating terminal output. Includes every fix from
1.1.1.

### Added
- **MCP server**: `parallelogram mcp` speaks the Model Context Protocol over
  stdio with three tools, `check_dataset`, `fix_dataset` and `list_rules`. No
  new dependencies. Every result carries the CLI's `exit_code`, and
  `fix_dataset` lists only the records it dropped or changed, so a large clean
  dataset is still a small result. Setup for Claude Code, Codex and Cursor is
  in [docs/agents.md](docs/agents.md).
- **Python API**: `from parallelogram import check, fix, rules`. Results have
  `exit_code` and `to_dict()`, which returns the same object as the CLI's
  `--json`. The CLI now runs on this API, so the two can't drift.
- **`check --json --limit N`**: lists the N most important issues (errors
  first) while every count stays exact. Adds `issues_truncated` and a `by_rule`
  breakdown to the JSON, still `schema_version: 1`.
- **`parallelogram rules [--json]`**: every rule with its severity, whether it
  is fixable, and what `--fix` does for it.
- `llms.txt` on parallelogram.dev, and a block to paste into your own
  `AGENTS.md` or `CLAUDE.md`.

### Changed
- `--fix --output` refuses to write over the input or eval file, including
  through symlinks and hard links, even with `--dry-run`.
- Read and write failures in `check` and `--fix` now exit 3 with a message on
  stderr, instead of a traceback, and never print a success JSON object.
  Invalid options still exit 2.

## 1.1.1 — 2026-11-07

Fixes from an independent review of 1.1.0. Every fix has a regression test.
Upgrade if you use `--eval`, preference or Alpaca data, or run in CI where logs
are shared.

### Security
- A credential inside a malformed field, such as a message `role`, was quoted in
  full by the schema diagnostic. All diagnostics and the report's role summary
  are now redacted centrally, even when the `secrets` rule is disabled.
- Real keys containing words like `your` or `example` inside them were treated
  as documentation placeholders and skipped. Placeholders now have to be a
  whole segment of the key.
- Google API keys ending in `-` were missed.

### Fixed
- `--eval`: the eval file's format is detected on its own instead of assumed to
  match the training file. An eval file with an invalid line now exits 3 with
  the line number instead of being partly ignored.
- `--fix` on Alpaca: redacting text in the instruction no longer merges the
  `input` field into it, and `null` optional fields stay `null`.
- Preference pairs keep their real prompt boundary through validation and
  `--fix`, so multi-turn `chosen` and `rejected` answers are checked, matched
  against `--eval` and deduplicated correctly. An empty `chosen` answer is now
  an error.
- `pii`: fewer false positives on arithmetic, coordinates, identifiers that
  embed digits and NANP-invalid numbers; subdomains of reserved example
  domains and `+1 555` numbers are ignored as documented.
- `--fix` rechecks a record after each text repair, so an issue uncovered by
  one fix (for example an email hidden behind a stray BOM) is repaired too.
- Records dropped by `eval-overlap` or `duplicates` keep the record of fixes
  already applied to them, and overlap decisions are made on the original text.
- Non-string `role` values are reported as schema errors instead of crashing.
- A ShareGPT-looking record that also has `messages` is no longer
  misdetected.
- Docs: format lists and exit codes now match the CLI.

## 1.1.0 — 2026-11-07

The biggest release since the first one: two new dataset formats, three new
checks that run by default, and train/eval leakage detection.

### Added
- **Alpaca format**: `instruction`, optional `input`, `output`, plus
  LLaMA-Factory's `system` and `history`. Auto-detected; `--fix --output`
  writes records back as Alpaca, restoring the instruction/input split and
  keeping extra fields such as `id`.
- **Preference (DPO) format**: `prompt` / `chosen` / `rejected` as strings or
  message lists, or TRL's implicit-prompt shape. Every rule now checks both
  sides of a pair: role order, empty turns and encoding on the rejected
  answer, context-window on the longer side, and duplicates that tell pairs
  apart by their rejected answer.
- **`secrets` rule** (error, fixable): API keys and tokens for OpenAI,
  Anthropic, AWS, GitHub, Hugging Face, Slack, Google and Stripe, and private
  key blocks. `--fix` redacts them. Matches are never printed in full, so CI
  logs don't leak them; documentation placeholders are ignored.
- **`pii` rule** (warning, fixable): email addresses, phone numbers and
  Luhn-valid card numbers. Reserved example domains and 555 numbers are
  ignored. `--fix` replaces them with typed placeholders.
- **`preference` rule** (error): pairs whose chosen and rejected answers are
  identical, which carry no training signal.
- **Train/eval overlap**: `check --eval eval.jsonl` and `report --eval`, plus
  an `eval` input on the GitHub Action. The `eval-overlap` rule flags train
  records whose content or prompt appears in the eval set; `--fix` drops them.

### Changed
- `--fix` credits records dropped by cross-record rules to the rule that
  dropped them; previously every such drop was reported as `duplicates`.

### Upgrade note
- New rules arrive in minor versions, as the stability contract allows. A
  dataset that passed 1.0 can now fail with a leaked credential (exit 2) or
  report PII as a warning (exit 1). Run `--fix` to redact both.

## 1.0.0 — 2026-11-06

### Added
- Eager `parallelogram --version` option, backed by the same version source
  used to build the package.
- `prompt-completion` format: string pairs, conversational message lists,
  mixed halves, per-record auto-detection, specific schema diagnostics,
  source-shape-preserving fixes, and an example dataset.
- Stable `schema_version: 1` JSON for check, fix, and report, with every field
  documented in `docs/json-output.md`. Unversioned 0.4.x report baselines
  remain supported.
- Composite GitHub Action installing from its own checkout, with Markdown
  job summaries, optional tokenizers and baselines, configurable warning
  gating, and integration tests.

### Changed
- Declared the 1.0 stability contract for commands, flags, rule ids, exit
  codes, and versioned JSON fields; new rules and formats may arrive in
  minor releases. Package classifier is now Production/Stable.

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
