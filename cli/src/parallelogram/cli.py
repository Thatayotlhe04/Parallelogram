"""parallelogram CLI.

Single command: `parallelogram check <path>`. Designed to be the
fastest possible local pre-flight before kicking off a training run.

Exit codes are deliberately minimal:
  0 — clean (no issues)
  1 — warnings only
  2 — errors

These map cleanly to CI gates without any extra wiring.

`--fix` (Phase 2 mechanical tier) attempts to mechanically repair fixable
issues — strip BOM, replace mojibake, drop empty turns, truncate
context-window overflow, deduplicate. SLM-tier fixes (rewrite broken role
sequences, fill in incomplete assistant turns) are not yet implemented and
will land in a future release as a paid hosted tier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core.fixer import Fixer, Disposition
from .core.io import atomic_write_jsonl
from .core.runner import Runner
from .core.rules import registry

# Importing the rule modules triggers self-registration via @registry.register.
from .rules import (  # noqa: F401
    schema,
    roles,
    empty_content,
    context_window,
    duplicates,
    encoding,
)
from .output.terminal import render_report
from .output.json_output import render_json


app = typer.Typer(
    name="parallelogram",
    help="Strict validator for fine-tuning datasets. Run before you train.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def check(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to JSONL dataset."),
    format: str = typer.Option(
        "openai-chat",
        "--format", "-f",
        help="Dataset format. Only 'openai-chat' is supported in v0.1.",
    ),
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer", "-t",
        help="HuggingFace tokenizer name (e.g. meta-llama/Llama-3-8B). Required for context-window check.",
    ),
    max_seq_len: int = typer.Option(
        4096,
        "--max-seq-len",
        help="Max tokens per record. Records over this are flagged because frameworks truncate them silently.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Write only error-free records to this file. With --fix, writes the repaired dataset.",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt mechanical fixes (dedupe, truncate, BOM strip, etc.). "
             "SLM-tier fixes are not yet available.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="With --fix, report what would change without writing any files.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a JSON report to stdout (suppresses pretty output).",
    ),
    disable: list[str] = typer.Option(
        [],
        "--disable",
        help="Disable a rule by id. Repeat to disable multiple. "
             "Disabling rules invalidates the exit-0 guarantee — see warnings.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
) -> None:
    """Check a fine-tuning dataset for problems that would silently corrupt training."""
    if format != "openai-chat":
        typer.echo(
            f"format {format!r} is not supported in v0.1 (only 'openai-chat').",
            err=True,
        )
        raise typer.Exit(2)

    if dry_run and not fix:
        typer.echo("--dry-run is only meaningful with --fix.", err=True)
        raise typer.Exit(2)

    # ── --disable handling — three layers of safety ───────────────────────
    #
    # 1. Reject unknown rule ids outright. Typos here are a foot-gun: the
    #    user thinks they disabled something but the run uses the full set.
    # 2. Refuse to disable schema. Every other rule assumes structurally
    #    valid records; disabling schema means the others silently no-op
    #    on malformed input, producing deceptively clean output.
    # 3. Loud stderr warning naming exactly what got disabled, plus an
    #    explicit note that the exit-0 guarantee no longer applies.
    known_rule_ids = {rc.id for rc in registry.all()}
    disabled = set(disable)
    unknown = disabled - known_rule_ids
    if unknown:
        typer.echo(
            f"unknown rule id(s) in --disable: {sorted(unknown)}. "
            f"Valid rule ids: {sorted(known_rule_ids)}",
            err=True,
        )
        raise typer.Exit(2)

    if "schema" in disabled:
        typer.echo(
            "the 'schema' rule cannot be disabled — every other rule "
            "depends on its guarantees about record structure.",
            err=True,
        )
        raise typer.Exit(2)

    if disabled:
        sorted_disabled = sorted(disabled)
        typer.echo(
            "  ! WARNING: --disable in effect for: "
            + ", ".join(sorted_disabled),
            err=True,
        )
        typer.echo(
            "    The 'if it exits 0, your run won't fail' guarantee "
            "applies only with all rules enabled.",
            err=True,
        )

    rule_classes = [rc for rc in registry.all() if rc.id not in disabled]

    rules = []
    for rc in rule_classes:
        if rc.id == "context-window":
            rules.append(rc({"tokenizer": tokenizer, "max_seq_len": max_seq_len}))
        else:
            rules.append(rc())

    runner = Runner(rules)
    console = Console(no_color=no_color)

    if fix:
        # ── Fix mode ────────────────────────────────────────────────────
        report, parsed, _, unparseable = runner.run_with_records(str(path))
        fixer = Fixer(rules)
        fix_report = fixer.fix(parsed, report.issues, unparseable)

        # Format and emit results.
        if json_output:
            payload = {
                "file": str(path),
                "mode": "fix",
                "dry_run": dry_run,
                "disabled_rules": sorted(disabled),
                "summary": {
                    "total": fix_report.total_records,
                    "unchanged": fix_report.unchanged,
                    "fixed": fix_report.fixed,
                    "dropped": fix_report.dropped,
                    "unparseable": fix_report.unparseable,
                    "emitted": len(fix_report.clean_records),
                },
                "fixes_by_rule": fix_report.fixes_by_rule,
                "outcomes": [
                    {
                        "line": o.line_no,
                        "disposition": o.disposition.value,
                        "rules_fixed": o.rules_fixed,
                        "rules_unfixable": o.rules_unfixable,
                    }
                    for o in fix_report.outcomes
                ],
            }
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            _render_fix_report(console, fix_report, str(path), dry_run,
                               has_output=output is not None,
                               disabled_rules=sorted(disabled))

        if not dry_run and output:
            lines = [json.dumps(rec, ensure_ascii=False)
                     for _, rec in fix_report.clean_records]
            atomic_write_jsonl(output, lines)
            if not json_output:
                console.print(f"  [green]→[/green] Wrote {len(lines)} records to {output}")
        elif not dry_run and not output:
            # Without --output, --fix does nothing destructive but we should
            # tell the user that no file was written.
            if not json_output:
                console.print(
                    "  [yellow]![/yellow] --fix was requested without --output. "
                    "Re-run with --output PATH to write the repaired dataset."
                )

        # Exit code semantics for fix mode:
        #   0 if everything is now clean (no dropped records)
        #   1 if some records were dropped (partial fix)
        #   2 if nothing was fixable (no records emitted)
        if not fix_report.clean_records:
            raise typer.Exit(2)
        if fix_report.dropped or fix_report.unparseable:
            raise typer.Exit(1)
        raise typer.Exit(0)

    # ── Check-only mode (existing behavior) ────────────────────────────
    report, clean = runner.run(str(path))

    if json_output:
        render_json(report, str(path), disabled_rules=sorted(disabled))
    else:
        render_report(report, console, str(path), disabled_rules=sorted(disabled))

    if output:
        lines = [raw if raw.endswith("\n") else raw + "\n" for _, raw in clean]
        atomic_write_jsonl(output, [l.rstrip("\n") for l in lines])
        if not json_output:
            console.print(
                f"  [green]→[/green] Wrote {len(clean)} clean records to {output}"
            )

    if report.has_errors:
        raise typer.Exit(2)
    if report.has_warnings:
        raise typer.Exit(1)
    raise typer.Exit(0)


def _render_fix_report(console, fr, path: str, dry_run: bool,
                       has_output: bool,
                       disabled_rules: list[str] | None = None) -> None:
    """Pretty-print the fix report for a human."""
    from rich.panel import Panel
    from rich.text import Text

    title = "parallelogram --fix" + (" (dry run)" if dry_run else "")

    # Per-rule fix counts
    if fr.fixes_by_rule:
        console.print()
        for rid, count in sorted(fr.fixes_by_rule.items()):
            console.print(f"  [green]✓[/green] [cyan]{rid}[/cyan] [dim]·[/dim] {count} fix{'es' if count != 1 else ''}")

    # Show first few dropped records so the user knows what was lost
    dropped_outcomes = [o for o in fr.outcomes if o.disposition == Disposition.DROPPED]
    if dropped_outcomes:
        console.print()
        console.print("  [red]✗[/red] [bold]dropped:[/bold]")
        for o in dropped_outcomes[:5]:
            if o.rules_unfixable:
                reason = ", ".join(o.rules_unfixable) + " (unfixable)"
            elif o.rules_fixed:
                reason = ", ".join(o.rules_fixed) + " (dropped as fix)"
            else:
                reason = "—"
            console.print(f"      [dim]{path}:[/dim]{o.line_no} [dim]→[/dim] {reason}")
        if len(dropped_outcomes) > 5:
            console.print(f"      [dim]… and {len(dropped_outcomes) - 5} more[/dim]")

    # Summary — appended note when rules were disabled, so a clean run
    # with disabled rules can never be mistaken for a clean full run.
    summary_parts = [
        (f"{fr.total_records} records  ", "bold"),
        (f"{fr.unchanged} unchanged  ", "dim"),
        (f"{fr.fixed} fixed  ", "green bold" if fr.fixed else "dim"),
        (f"{fr.dropped} dropped  ", "red bold" if fr.dropped else "dim"),
        (f"{fr.unparseable} unparseable", "red bold" if fr.unparseable else "dim"),
    ]
    if disabled_rules:
        summary_parts.append((f"  ({len(disabled_rules)} rule(s) disabled)", "yellow"))
    summary = Text.assemble(*summary_parts)
    border = "green" if not (fr.dropped or fr.unparseable or disabled_rules) else "yellow"
    console.print()
    console.print(Panel.fit(summary, title=title, border_style=border))


if __name__ == "__main__":
    app()
