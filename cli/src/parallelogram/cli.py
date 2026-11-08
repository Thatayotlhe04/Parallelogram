"""parallelogram CLI.

Local pre-flight validation plus discovery and agent tools:

  parallelogram check <path>   — line-by-line diagnostics (+ --fix)
  parallelogram report <path>  — aggregate dataset health, for humans and CI
  parallelogram rules          — discover rule metadata and repair behavior
  parallelogram mcp            — serve local dataset tools over stdio

Exit codes are deliberately minimal and stable:
  0 — clean (no issues)
  1 — warnings only
  2 — errors
  3 — operational failure such as unreadable or invalid --eval input (check,
      fix, report), or quality regression vs --baseline (report)

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
from rich.markup import escape

from . import __version__, api
from .api import ParallelogramError, _build_rules
from .core.fixer import Disposition
from .core.report import Severity
from .formats import supported_formats
from .output.terminal import render_report


app = typer.Typer(
    name="parallelogram",
    help="Strict validator for fine-tuning datasets. Run before you train.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"parallelogram {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Strict validator for fine-tuning datasets. Run before you train."""


@app.command()
def check(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to JSONL dataset."),
    dataset_format: str = typer.Option(
        "auto",
        "--format", "-f",
        help="Dataset format: 'auto' (default), 'openai-chat', 'sharegpt', 'prompt-completion', 'alpaca', or 'preference'.",
    ),
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer", "-t",
        help="Model or tokenizer for the context-window check — an OpenAI model "
             "(gpt-4o), an HF repo or alias (Qwen/Qwen2.5-7B, mistral, llama-3). "
             "Omit for an approximate length-based count (reported as warnings).",
    ),
    max_seq_len: int = typer.Option(
        4096,
        "--max-seq-len",
        help="Max tokens per record. Records over this are flagged because frameworks truncate them silently.",
    ),
    eval_path: Optional[Path] = typer.Option(
        None,
        "--eval",
        exists=True, dir_okay=False, readable=True,
        help="Eval set to check for overlap: flags train records whose content or prompt also appears in it.",
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
    limit: Optional[int] = typer.Option(
        None, "--limit", min=0, help="Maximum JSON issues to list; counts remain exact.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
) -> None:
    """Check a fine-tuning dataset for problems that would silently corrupt training."""
    if dry_run and not fix:
        typer.echo("--dry-run is only meaningful with --fix.", err=True)
        raise typer.Exit(2)
    if fix and limit is not None:
        typer.echo("--limit applies to check issues, not --fix outcomes.", err=True)
        raise typer.Exit(2)

    try:
        options = {"format": dataset_format, "tokenizer": tokenizer,
                   "max_seq_len": max_seq_len, "eval_path": eval_path, "disable": disable,
                   "output": output}
        result = (api.fix(path, dry_run=dry_run, **options) if fix
                  else api.check(path, limit=limit, **options))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    except ParallelogramError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(3)

    if result.disabled_rules:
        typer.echo("  ! WARNING: --disable in effect for: "
                   + ", ".join(result.disabled_rules), err=True)
        typer.echo("    The 'if it exits 0, your run won't fail' guarantee "
                   "applies only with all rules enabled.", err=True)

    console = Console(no_color=no_color)
    if json_output:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif fix:
        _render_fix_report(console, result.report, str(path), dry_run,
                           has_output=output is not None,
                           disabled_rules=result.disabled_rules)
    else:
        render_report(result.report, console, str(path),
                      disabled_rules=result.disabled_rules)

    if fix:
        for err in result.report.fixer_errors:
            typer.echo(f"  ! FIXER ERROR: {err}", err=True)
        if not json_output and not dry_run:
            if output:
                console.print(f"  [green]→[/green] Wrote {len(result.report.clean_records)} records to {escape(str(output))}")
            else:
                console.print("  [yellow]![/yellow] --fix was requested without --output. "
                              "Re-run with --output PATH to write the repaired dataset.")
    elif output and not json_output:
        clean_line_nos = {ln for ln, _ in result.clean_records}
        warned_lines = {i.line_no for i in result.report.issues
                        if i.severity == Severity.WARNING and i.line_no in clean_line_nos}
        console.print(f"  [green]→[/green] Wrote {len(result.clean_records)} error-free records to {escape(str(output))}")
        if warned_lines:
            console.print(f"  [yellow]![/yellow] {len(warned_lines)} of them still carry "
                          "warnings (e.g. mojibake/BOM) and were copied verbatim — "
                          "re-run with [bold]--fix --output[/bold] to repair them.")
    raise typer.Exit(result.exit_code)


@app.command("rules")
def list_rules(json_output: bool = typer.Option(False, "--json", help="Emit rule metadata as JSON.")) -> None:
    """List validation rules and what mechanical fixes do."""
    metadata = api.rules()
    if json_output:
        typer.echo(json.dumps(metadata, indent=2))
    else:
        for rule in metadata:
            typer.echo(f"{rule['id']} ({rule['severity']}; fixable={str(rule['fixable']).lower()}; "
                       f"fix_scope={rule['fix_scope']}): {rule['description']}")
            typer.echo(f"  --fix: {rule['fix_description']}")


@app.command()
def mcp(
    root: Optional[Path] = typer.Option(
        None, "--root",
        help="Directory fix_dataset may write into (default: the current directory). "
             "Output paths outside it are refused.",
        exists=True, file_okay=False, dir_okay=True,
    ),
) -> None:
    """Serve local dataset tools over newline-delimited JSON-RPC on stdio."""
    from .mcp_server import serve
    serve(root=root)


@app.command()
def report(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to JSONL dataset."),
    dataset_format: str = typer.Option(
        "auto",
        "--format", "-f",
        help="Dataset format: 'auto' (default), 'openai-chat', 'sharegpt', 'prompt-completion', 'alpaca', or 'preference'.",
    ),
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer", "-t",
        help="Model or tokenizer for token-risk stats — an OpenAI model (gpt-4o) "
             "or an HF repo/alias (Qwen/Qwen2.5-7B, mistral, llama-3). "
             "Omit for an approximate count, labeled as estimated.",
    ),
    max_seq_len: int = typer.Option(
        4096,
        "--max-seq-len",
        help="Token budget per record for the risk stats.",
    ),
    eval_path: Optional[Path] = typer.Option(
        None,
        "--eval",
        exists=True, dir_okay=False, readable=True,
        help="Eval set to check for overlap: flags train records whose content or prompt also appears in it.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the report as JSON to stdout (machine-readable; also the "
             "baseline format for --baseline).",
    ),
    markdown: bool = typer.Option(
        False,
        "--markdown",
        help="Emit the report as GitHub-flavored Markdown "
             "(pipe into $GITHUB_STEP_SUMMARY or a PR comment).",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Also write the report (in the chosen format) to this file.",
    ),
    baseline: Optional[Path] = typer.Option(
        None,
        "--baseline",
        help="Path to a previous `report --json` file. If dataset quality "
             "regressed against it (rate-based, so growth alone never "
             "fails), exit 3.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
) -> None:
    """Summarize dataset health: clean/error/warning counts, what --fix would
    do, token risk, duplicate clusters, and shape — for humans and CI.

    Exit codes: 0 clean · 1 warnings · 2 errors · 3 invalid eval input or regression vs --baseline.
    """
    from .core.summary import build_summary, compare_to_baseline
    from .output.summary_output import render_terminal_summary, render_markdown_summary

    if dataset_format not in supported_formats():
        typer.echo(
            f"format {dataset_format!r} is not supported. "
            f"Valid formats: {supported_formats()}",
            err=True,
        )
        raise typer.Exit(2)
    if json_output and markdown:
        typer.echo("--json and --markdown are mutually exclusive.", err=True)
        raise typer.Exit(2)

    try:
        rules = _build_rules(set(), tokenizer, max_seq_len, eval_path, dataset_format)
    except (ParallelogramError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(3)

    summary = build_summary(str(path), rules, dataset_format=dataset_format)

    # ── render ──────────────────────────────────────────────────────────
    if json_output:
        rendered = json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n"
        sys.stdout.write(rendered)
    elif markdown:
        rendered = render_markdown_summary(summary)
        sys.stdout.write(rendered)
    else:
        rendered = render_markdown_summary(summary)  # files always get md/json
        render_terminal_summary(summary, Console(no_color=no_color))

    if out:
        if json_output:
            out.write_text(json.dumps(summary.to_dict(), indent=2,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            out.write_text(rendered, encoding="utf-8")
        typer.echo(f"report written to {out}", err=True)

    # ── baseline gate ───────────────────────────────────────────────────
    if baseline is not None:
        try:
            base = json.loads(baseline.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            typer.echo(f"could not read baseline {baseline}: {exc}", err=True)
            raise typer.Exit(2)
        regressions = compare_to_baseline(summary, base)
        if regressions:
            typer.echo("dataset quality regressed vs baseline:", err=True)
            for r in regressions:
                typer.echo(f"  ✗ {r}", err=True)
            raise typer.Exit(3)
        typer.echo("no regression vs baseline.", err=True)

    raise typer.Exit(summary.exit_code)


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
            console.print(f"      [dim]{escape(str(path))}:[/dim]{o.line_no} [dim]→[/dim] {escape(str(reason))}")
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


def run() -> None:
    """Console entry point. An unexpected crash must never look like success.

    Python exits 1 on an uncaught exception, and 1 means "warnings only", so a
    crash used to read as a pass to any gate that tolerates warnings (the
    GitHub Action does by default). Unexpected failures exit 3 instead, the
    operational-failure code, with a one-line message and no traceback.
    """
    try:
        app()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 — last line of defence
        from .core.report import scrub

        sys.stderr.write(scrub(f"parallelogram: internal error: {type(exc).__name__}: {exc}") + "\n")
        raise SystemExit(3)


if __name__ == "__main__":
    run()
