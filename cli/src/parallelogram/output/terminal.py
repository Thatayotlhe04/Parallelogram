"""Rich terminal output for the human at a terminal.

Layout goal: a developer scrolling output in their terminal should see, for
each issue, the file:line, rule id, and a one-sentence reason — without their
eyes having to scan past decoration.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..core.report import Report, Severity


def render_report(
    report: Report,
    console: Console | None = None,
    file_path: str = "",
    disabled_rules: list[str] | None = None,
) -> None:
    if console is None:
        console = Console()

    disabled_rules = disabled_rules or []

    if report.is_clean:
        # When rules are disabled the run is "clean (with caveats)" — never
        # let a disabled-rules run be visually indistinguishable from a
        # full-rules pass. The exit-0 guarantee covers only full runs.
        if disabled_rules:
            console.print(Panel.fit(
                Text.assemble(
                    ("✓ ", "bold yellow"),
                    f"Clean: {report.total_records} records validated  ",
                    (f"({len(disabled_rules)} rule(s) disabled: "
                     f"{', '.join(disabled_rules)})", "yellow"),
                ),
                title="parallelogram",
                border_style="yellow",
            ))
        else:
            console.print(Panel.fit(
                Text.assemble(
                    ("✓ ", "bold green"),
                    f"Clean: {report.total_records} records validated",
                ),
                title="parallelogram",
                border_style="green",
            ))
        return

    by_line: dict[int | None, list] = {}
    for issue in report.issues:
        by_line.setdefault(issue.line_no, []).append(issue)

    # None-keyed (global) issues first, then sorted by line.
    keys = sorted(by_line.keys(), key=lambda x: (x is not None, x or 0))
    for key in keys:
        for issue in by_line[key]:
            tag_color = "red" if issue.severity == Severity.ERROR else "yellow"
            tag = "✗" if issue.severity == Severity.ERROR else "!"
            location = f"{file_path}:{issue.line_no}" if issue.line_no else "(global)"
            console.print(
                Text.assemble(
                    (f"  {tag} ", f"bold {tag_color}"),
                    (f"{location} ", "dim"),
                    (f"[{issue.rule_id}] ", "cyan"),
                    issue.message,
                )
            )
            if issue.detail:
                console.print(f"      [dim]→ {issue.detail}[/dim]")

    n_err = len(report.errors)
    n_warn = len(report.warnings)

    console.print()
    summary_parts = [
        (f"{report.total_records} records  ", "bold"),
        (f"{report.valid_records} clean  ", "green"),
        (f"{n_err} errors  ", "red bold" if n_err else "dim"),
        (f"{n_warn} warnings", "yellow bold" if n_warn else "dim"),
    ]
    if disabled_rules:
        summary_parts.append((f"  ({len(disabled_rules)} rule(s) disabled)", "yellow"))
    summary = Text.assemble(*summary_parts)
    border = "red" if n_err else "yellow"
    console.print(Panel.fit(summary, title="parallelogram", border_style=border))
