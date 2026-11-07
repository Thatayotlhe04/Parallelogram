# JSON output contract — schema_version 1

`check --json`, `check --fix --json` (including `--dry-run`), and `report --json`
write one JSON object to stdout. Diagnostics go to stderr. Each object contains
`schema_version: 1` (integer) and `file` (string, the input path as supplied to the
CLI). Object key order and diagnostic prose are not a contract. Map keys such as
rule ids and format ids can grow as new rules and formats are added. The fields
and their meanings below are stable through 1.x; removing or renaming a field
requires 2.0. Consumers should tolerate additional fields.

Counts are nonnegative integers unless a different type is stated. Blank lines
are ignored. Line numbers are one-based physical file lines. Informational
issues never affect the process exit code. No dataset content is included as a
separate field, though diagnostic messages can quote malformed values with detected secrets redacted.

If `--eval` cannot be read or contains invalid JSON/schema, commands abort before
producing a JSON payload, write a diagnostic to stderr, and exit 3. An empty eval
file is valid. The eval format is auto-detected independently of the train format.

## `check --json`

| Field | Type and meaning |
|---|---|
| `schema_version` | Integer, always `1`. |
| `file` | String, dataset path. |
| `total_records` | Nonblank input lines, including invalid JSON. |
| `valid_records` | Records without errors; records with warnings still count. |
| `disabled_rules` | Array of disabled rule-id strings, sorted; empty by default. |
| `summary.errors` | Number of error issues, not distinct records. |
| `summary.warnings` | Number of warning issues. |
| `summary.clean` | Boolean, true only when there are no issues at any severity, including info. Consequently an informational tokenizer note can make this false while the command exits 0. |
| `issues` | Array of issue objects, including informational notes. |
| `issues[].rule` | String rule id. Current ids: `schema`, `roles`, `empty-content`, `context-window`, `duplicates`, `encoding`, `secrets`, `pii`, `preference`, `eval-overlap`. |
| `issues[].severity` | String: `error`, `warning`, or `info`. |
| `issues[].line` | Integer line number, or null for dataset-wide notes. |
| `issues[].message` | String diagnostic; wording is not stable. |
| `issues[].detail` | String additional explanation, or null; wording is not stable. |
| `issues[].fixable` | Boolean, whether a mechanical fix can be attempted; not a guarantee the record will survive revalidation. |

Exit: 2 with any errors, otherwise 1 with warnings, otherwise 0.

## `check --fix --json`

This mode replaces the check diagnostics object with a fix result object.

| Field | Type and meaning |
|---|---|
| `schema_version` | Integer, always `1`. |
| `file` | String, dataset path. |
| `mode` | String, always `fix`. |
| `dry_run` | Boolean, whether writing was suppressed by `--dry-run`. |
| `disabled_rules` | Array of disabled rule-id strings, sorted; empty by default. |
| `summary.total` | Nonblank input lines. |
| `summary.unchanged` | Surviving records that needed no applied repair. |
| `summary.fixed` | Surviving records with repairs applied. |
| `summary.dropped` | Parsed records removed by deduplication or unfixable errors. |
| `summary.unparseable` | Lines that were not valid JSON; separate from dropped. |
| `summary.emitted` | Records available for output: unchanged + fixed, including in dry-run mode. |
| `fixes_by_rule` | Object mapping rule-id strings to counts of applied fixes (not necessarily distinct records). Empty when none were applied. |
| `fixer_errors` | Array of strings describing unexpected fixer exceptions; empty normally. |
| `outcomes` | Array of per-line objects, sorted by line number. |
| `outcomes[].line` | Integer physical line number. |
| `outcomes[].disposition` | String: `unchanged`, `fixed`, `dropped`, or `unparseable`. |
| `outcomes[].rules_fixed` | Array of rule-id strings for applied repairs; may contain repeated ids. Deduplicated records have disposition `dropped` and `duplicates` here. |
| `outcomes[].rules_unfixable` | Array of rule-id strings that prevented repair; may contain repeated ids. |

Exit: 2 when nothing can be emitted; otherwise 1 if any records were dropped,
any lines were unparseable, or a fixer exception occurred; otherwise 0.

## `report --json`

`--out PATH` saves the same object. A report is also the input to `--baseline`.
Baselines from 0.4.x without `schema_version` remain accepted: comparison reads
the `rates` object, skipping absent individual rates.

| Field | Type and meaning |
|---|---|
| `schema_version` | Integer, always `1`. |
| `file` | String, dataset path. |
| `format` | String requested parser: `auto`, `openai-chat`, `sharegpt`, `prompt-completion`, `alpaca`, or `preference`. |
| `disabled_rules` | Array of disabled rule-id strings; currently always empty for the report command. |
| `summary.total_records` | Nonblank input lines, including invalid JSON. |
| `summary.clean_records` | Records without errors; may have warnings (same definition as check's `valid_records`). |
| `summary.records_with_errors` | Distinct lines with errors. |
| `summary.records_with_warnings` | Distinct lines with warnings; may overlap error lines. |
| `summary.unparseable_records` | Lines with invalid JSON; also counted among error lines. |
| `summary.error_count` | Number of error issues. |
| `summary.warning_count` | Number of warning issues. |
| `summary.fixable_issue_count` | Number of issues marked mechanically fixable. |
| `summary.exit_code` | Integer dataset status: 0 clean, 1 warnings, 2 errors. This is computed before baseline comparison, so the process may exit 3 while this field remains 0, 1, or 2. |
| `issues_by_rule` | Object keyed by rule id, only for rules that emitted issues (including info-only rules). |
| `issues_by_rule.<id>.errors` | Error issue count for this rule. |
| `issues_by_rule.<id>.warnings` | Warning issue count for this rule. |
| `issues_by_rule.<id>.fixable` | Mechanically fixable issue count for this rule. |
| `fix_projection.unchanged` | Records a dry fix pass would retain unchanged. |
| `fix_projection.fixed` | Records it would retain after repairs. |
| `fix_projection.dropped` | Parsed records it would drop; excludes unparseable lines. |
| `fix_projection.unparseable` | Invalid JSON lines. |
| `fix_projection.emitted` | Unchanged + fixed. |
| `token_stats.counting` | String: `exact` or `estimated`. |
| `token_stats.method` | String identifying the counting method; not a fixed enumeration. |
| `token_stats.max_seq_len` | Integer token budget from the option. |
| `token_stats.over_budget` | Counted records exceeding that budget. |
| `token_stats.at_risk` | Counted records between 85% and 100% of the budget, inclusive. |
| `token_stats.median_tokens` | Integer median of countable records, fractional medians truncated; 0 if none. |
| `token_stats.max_tokens` | Maximum token count, or 0 if none. |
| `duplicate_stats.clusters` | Number of groups with at least two equivalent records. |
| `duplicate_stats.duplicate_records` | Redundant copies across clusters, excluding each cluster's first record. |
| `duplicate_stats.largest_cluster` | Largest duplicate cluster size including its first record; 0 if none. |
| `format_breakdown.declared_format` | String, same requested parser as `format`. |
| `format_breakdown.detected_formats` | Object mapping source-format strings to counts of parsed lines, including schema-invalid records but excluding invalid JSON. |
| `format_breakdown.roles` | Object mapping normalized role strings to message counts. Non-string/missing roles count under `invalid`; unknown string roles retain their names with detected secrets redacted. |
| `format_breakdown.turns_per_record.min` | Minimum message-list length, or 0 if none. |
| `format_breakdown.turns_per_record.median` | Integer median message-list length, fractional medians truncated; 0 if none. |
| `format_breakdown.turns_per_record.max` | Maximum message-list length, or 0 if none. |
| `format_breakdown.ends_on_assistant` | Number of normalized message lists ending with an assistant role. |
| `rates.clean_fraction` | Number: clean_records / max(1, total_records). |
| `rates.errors_per_record` | Number: error_count / max(1, total_records); can exceed 1. |
| `rates.duplicates_per_record` | Number: duplicate_records / max(1, total_records). |
| `rates.over_budget_per_record` | Number: over_budget / max(1, total_records). |
| `rates.dropped_per_record` | Number: fix_projection.dropped / max(1, total_records); excludes unparseable lines. |

Token and shape statistics include parsed records that have a countable
`messages` list, even if validation finds other errors. `token_stats` is empty
when the Python summary API is used without the context-window rule; the CLI
always includes that rule.

Exit: the dataset status above, or 3 if baseline rates regress. Increasing any
of the four non-clean rates or decreasing `clean_fraction` by more than 1e-9 is a
regression. An unreadable/invalid JSON baseline exits 2; a baseline without a
`rates` object is reported by the regression gate and exits 3.
