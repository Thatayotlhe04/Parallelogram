"""Silent Python entry points shared by the CLI and local agent tools."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

from .core.fixer import Fixer, FixReport
from .core.io import atomic_write_jsonl, atomic_write_bytes
from .core.report import Report
from .core.runner import Runner
from .core.rules import Rule, registry
from .formats import supported_formats
from .formats.sharegpt import to_sharegpt
from .formats.prompt_completion import to_prompt_completion
from .formats.alpaca import to_alpaca
from .formats.preference import to_preference
from .output.json_output import report_to_dict

# Import every built-in rule once to populate the registry, without importing CLI.
from .rules import (  # noqa: F401
    schema,
    roles,
    tool_calls,
    empty_content,
    context_window,
    duplicates,
    encoding,
    preference,
    secrets,
    pii,
    eval_overlap,
)
from .rules.eval_overlap import EvalInputError

_WRITERS = {
    "sharegpt": to_sharegpt,
    "prompt-completion": to_prompt_completion,
    "alpaca": to_alpaca,
    "preference": to_preference,
}


def _build_rules(disabled: set[str], tokenizer: str | None, max_seq_len: int,
                 eval_path: str | Path | None, dataset_format: str) -> list[Rule]:
    """Instantiate the enabled rules. eval-overlap only runs with --eval."""
    # Match CLI path validation even if the train file is empty or overlap disabled.
    if eval_path is not None:
        with Path(eval_path).open("rb"):
            pass
    rules = []
    for rc in registry.all():
        if rc.id in disabled:
            continue
        if rc.id == "context-window":
            rules.append(rc({"tokenizer": tokenizer, "max_seq_len": max_seq_len}))
        elif rc.id == "eval-overlap":
            if eval_path is not None:
                # The eval set's format is detected on its own; an invalid eval
                # file is an operational failure, not a finding.
                rule = rc({"eval_path": str(eval_path)})
                try:
                    rule._load()
                except EvalInputError as exc:
                    raise ParallelogramError(str(exc)) from exc
                rules.append(rule)
        else:
            rules.append(rc())
    return rules


class ParallelogramError(Exception):
    """A dataset operation failed (CLI exit 3), rather than finding data issues."""


def _validate_options(format: str, disable: Iterable[str], limit: int | None) -> set[str]:
    if format not in supported_formats():
        raise ValueError(f"format {format!r} is not supported. Valid formats: {supported_formats()}")
    if isinstance(disable, str):
        raise ValueError("disable must be an iterable of rule ids, not a string")
    disabled = set(disable)
    known = {rc.id for rc in registry.all()}
    unknown = disabled - known
    if unknown:
        raise ValueError(f"unknown rule id(s) in --disable: {sorted(unknown)}. Valid rule ids: {sorted(known)}")
    if "schema" in disabled:
        raise ValueError("the 'schema' rule cannot be disabled — every other rule depends on its guarantees about record structure.")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    return disabled


def _refuse_input_output(path: str | Path, output: str | Path | None,
                         eval_path: str | Path | None) -> None:
    if output is None:
        return
    destination = Path(output).resolve()
    for source in (path, eval_path):
        if source is not None and (destination == Path(source).resolve()
                                  or (destination.exists() and Path(source).exists()
                                      and destination.samefile(source))):
            raise ValueError("output must be different from the input and eval paths")


@dataclass
class CheckResult:
    """Complete validation evidence; limit affects serialization only."""
    report: Report
    file: str
    disabled_rules: list[str]
    clean_records: list[tuple[int, str]]
    limit: int | None = None

    @property
    def exit_code(self) -> int:
        return 2 if self.report.has_errors else 1 if self.report.has_warnings else 0

    def to_dict(self) -> dict:
        return report_to_dict(self.report, self.file, self.disabled_rules, self.limit)


@dataclass
class FixResult:
    """The same fix projection and status returned by check --fix --json."""
    report: FixReport
    file: str
    disabled_rules: list[str]
    dry_run: bool

    @property
    def exit_code(self) -> int:
        if not self.report.clean_records:
            return 2
        return 1 if (self.report.dropped or self.report.unparseable
                     or self.report.fixer_errors) else 0

    def to_dict(self) -> dict:
        fr = self.report
        return {
            "schema_version": 1,
            "file": self.file,
            "mode": "fix",
            "dry_run": self.dry_run,
            "disabled_rules": self.disabled_rules.copy(),
            "summary": {
                "total": fr.total_records,
                "unchanged": fr.unchanged,
                "fixed": fr.fixed,
                "dropped": fr.dropped,
                "unparseable": fr.unparseable,
                "emitted": len(fr.clean_records),
            },
            "fixes_by_rule": fr.fixes_by_rule.copy(),
            "fixer_errors": fr.fixer_errors.copy(),
            "outcomes": [{
                "line": o.line_no,
                "disposition": o.disposition.value,
                "rules_fixed": o.rules_fixed.copy(),
                "rules_unfixable": o.rules_unfixable.copy(),
            } for o in fr.outcomes],
        }


def check(path: str | Path, *, format: str = "auto",
          eval_path: str | Path | None = None, disable: Iterable[str] = (),
          tokenizer: str | None = None, max_seq_len: int = 4096,
          limit: int | None = None, output: str | Path | None = None) -> CheckResult:
    """Validate all records silently; optionally copy error-free source lines.

    Invalid options raise ValueError; operational failures raise
    ParallelogramError. Neither errors nor warnings in data raise exceptions.
    """
    disabled = _validate_options(format, disable, limit)
    try:
        _refuse_input_output(path, output, eval_path)
        enabled = _build_rules(disabled, tokenizer, max_seq_len, eval_path, format)
        report, clean = Runner(enabled, dataset_format=format).run(str(path))
        if output is not None:
            atomic_write_jsonl(Path(output), [raw.rstrip("\n") for _, raw in clean])
        return CheckResult(report, str(path), sorted(disabled), clean, limit)
    except (OSError, UnicodeError) as exc:
        raise ParallelogramError(f"Could not check {path}: {exc}") from exc


def fix(path: str | Path, *, output: str | Path | None = None,
        dry_run: bool = False, format: str = "auto",
        eval_path: str | Path | None = None, disable: Iterable[str] = (),
        tokenizer: str | None = None, max_seq_len: int = 4096) -> FixResult:
    """Repair into a separate file, or return a projection when output is omitted.

    A dry run never writes. Input/eval aliases are refused even in a dry run,
    so a preview can safely be repeated with dry_run=False.
    """
    disabled = _validate_options(format, disable, None)
    try:
        _refuse_input_output(path, output, eval_path)
        enabled = _build_rules(disabled, tokenizer, max_seq_len, eval_path, format)
        runner = Runner(enabled, dataset_format=format)
        report, parsed, _, unparseable = runner.run_with_records(str(path))
        fr = Fixer(enabled).fix(parsed, report.issues, unparseable)
        if output is not None and not dry_run:
            def to_disk(line_no, record):
                writer = _WRITERS.get(runner.source_formats.get(line_no, format))
                return writer(record, runner.source_metadata.get(line_no)) if writer else record

            if fr.unchanged == fr.total_records and not fr.fixer_errors:
                # A no-op fix preserves whitespace, CRLF and the final newline too.
                atomic_write_bytes(Path(output), Path(path).read_bytes())
            else:
                originals = dict(parsed)
                raw_lines = Path(path).read_bytes().splitlines(keepends=True)
                lines = []
                for line_no, record in fr.clean_records:
                    if record == originals.get(line_no):
                        raw = raw_lines[line_no - 1]
                        lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
                    else:
                        lines.append((json.dumps(to_disk(line_no, record), ensure_ascii=False) + "\n").encode("utf-8"))
                atomic_write_bytes(Path(output), b"".join(lines))
        return FixResult(fr, str(path), sorted(disabled), dry_run)
    except (OSError, UnicodeError) as exc:
        raise ParallelogramError(f"Could not fix {path}: {exc}") from exc


def rules() -> list[dict]:
    """Discover rule behavior directly from the registered rule classes."""
    return [{
        "id": rc.id,
        "severity": rc.severity.value,
        "fixable": rc.fixable,
        "fix_scope": rc.fix_scope,
        "description": rc.description,
        "fix_description": rc.fix_description,
    } for rc in registry.all()]
