"""Renderers for `parallelogram report` — terminal, and Markdown.

(JSON is `DatasetSummary.to_dict()` dumped directly in the CLI; there is
nothing to format.) The Markdown renderer targets GitHub: pipe it into
`$GITHUB_STEP_SUMMARY` or a PR comment and the tables just work.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.summary import DatasetSummary


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "—"


def render_terminal_summary(s: DatasetSummary, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    n = s.total_records

    # ── headline panel ─────────────────────────────────────────────────
    head = Text.assemble(
        (f"{n} records  ", "bold"),
        (f"{s.clean_records} clean ({_pct(s.clean_records, n)})  ",
         "green bold" if s.clean_records == n else "green"),
        (f"{s.records_with_errors} with errors  ",
         "red bold" if s.records_with_errors else "dim"),
        (f"{s.records_with_warnings} with warnings  ",
         "yellow" if s.records_with_warnings else "dim"),
        (f"{s.unparseable_records} unparseable",
         "red bold" if s.unparseable_records else "dim"),
    )
    border = "red" if s.exit_code == 2 else "yellow" if s.exit_code == 1 else "green"
    title = "parallelogram report"
    if s.disabled_rules:
        title += f" ({len(s.disabled_rules)} rule(s) disabled)"
        border = "yellow"
    console.print()
    console.print(Panel.fit(head, title=title, border_style=border))

    # ── issues by rule ─────────────────────────────────────────────────
    if s.issues_by_rule:
        t = Table(title=None, show_edge=False, pad_edge=False, box=None)
        t.add_column("rule", style="cyan")
        t.add_column("errors", justify="right")
        t.add_column("warnings", justify="right")
        t.add_column("fixable", justify="right", style="green")
        for rid, c in s.issues_by_rule.items():
            t.add_row(
                rid,
                Text(str(c["errors"]), style="red bold" if c["errors"] else "dim"),
                Text(str(c["warnings"]), style="yellow" if c["warnings"] else "dim"),
                str(c["fixable"]),
            )
        console.print()
        console.print(t)

    # ── fix projection ─────────────────────────────────────────────────
    fp = s.fix_projection
    if fp:
        console.print()
        console.print(Text.assemble(
            ("--fix would emit ", "dim"),
            (f"{fp.get('emitted', 0)}", "green bold"),
            (" records: ", "dim"),
            (f"{fp.get('unchanged', 0)} unchanged · ", "dim"),
            (f"{fp.get('fixed', 0)} fixed · ",
             "green" if fp.get("fixed") else "dim"),
            (f"{fp.get('dropped', 0)} dropped · ",
             "red" if fp.get("dropped") else "dim"),
            (f"{fp.get('unparseable', 0)} unparseable",
             "red" if fp.get("unparseable") else "dim"),
        ))

    # ── token risk ─────────────────────────────────────────────────────
    ts = s.token_stats
    if ts:
        exact = ts.get("counting") == "exact"
        console.print()
        console.print(Text.assemble(
            ("token budget ", "dim"),
            (f"{ts.get('max_seq_len')}", "bold"),
            ("  ·  counting: ", "dim"),
            (ts.get("counting", "?"),
             "cyan bold" if exact else "magenta"),
            (f" ({ts.get('method', '')})", "dim"),
        ))
        console.print(Text.assemble(
            ("  over budget: ", "dim"),
            (f"{ts.get('over_budget', 0)}",
             ("red bold" if exact else "yellow bold") if ts.get("over_budget") else "dim"),
            ("  at risk (≥85%): ", "dim"),
            (f"{ts.get('at_risk', 0)}", "yellow" if ts.get("at_risk") else "dim"),
            ("  median: ", "dim"), (f"{ts.get('median_tokens', 0)}", ""),
            ("  max: ", "dim"), (f"{ts.get('max_tokens', 0)}", ""),
        ))

    # ── duplicates ─────────────────────────────────────────────────────
    ds = s.duplicate_stats
    if ds:
        console.print(Text.assemble(
            ("duplicate clusters: ", "dim"),
            (f"{ds.get('clusters', 0)}", "red bold" if ds.get("clusters") else "dim"),
            ("  duplicate records: ", "dim"),
            (f"{ds.get('duplicate_records', 0)}",
             "red" if ds.get("duplicate_records") else "dim"),
            ("  largest cluster: ", "dim"),
            (f"{ds.get('largest_cluster', 0)}", "dim"),
        ))

    # ── shape ──────────────────────────────────────────────────────────
    fb = s.format_breakdown
    if fb:
        roles = "  ".join(f"{k}:{v}" for k, v in fb.get("roles", {}).items())
        detected = "  ".join(
            f"{k}:{v}" for k, v in fb.get("detected_formats", {}).items()
        )
        tp = fb.get("turns_per_record", {})
        console.print(Text.assemble(
            ("format: ", "dim"), (fb.get("declared_format", "?"), "cyan"),
            ("  detected: ", "dim"), (detected or "—", ""),
            ("  roles: ", "dim"), (roles or "—", ""),
            ("  turns/record: ", "dim"),
            (f"{tp.get('min', 0)}/{tp.get('median', 0)}/{tp.get('max', 0)} (min/med/max)", ""),
            ("  end on assistant: ", "dim"),
            (f"{fb.get('ends_on_assistant', 0)}/{n}", ""),
        ))
    console.print()


def render_markdown_summary(s: DatasetSummary) -> str:
    """GitHub-flavored Markdown — for $GITHUB_STEP_SUMMARY or PR comments."""
    n = s.total_records
    status = {0: "✅ clean", 1: "⚠️ warnings", 2: "❌ errors"}.get(s.exit_code, "?")
    lines = [
        f"## parallelogram report — `{s.file}`",
        "",
        f"**{status}** · exit code `{s.exit_code}`"
        + (f" · {len(s.disabled_rules)} rule(s) disabled: "
           f"`{', '.join(s.disabled_rules)}`" if s.disabled_rules else ""),
        "",
        "| records | clean | with errors | with warnings | unparseable |",
        "|---:|---:|---:|---:|---:|",
        f"| {n} | {s.clean_records} ({_pct(s.clean_records, n)}) "
        f"| {s.records_with_errors} | {s.records_with_warnings} "
        f"| {s.unparseable_records} |",
    ]

    if s.issues_by_rule:
        lines += [
            "",
            "### Issues by rule",
            "",
            "| rule | errors | warnings | fixable |",
            "|---|---:|---:|---:|",
        ]
        for rid, c in s.issues_by_rule.items():
            lines.append(f"| `{rid}` | {c['errors']} | {c['warnings']} | {c['fixable']} |")

    fp = s.fix_projection
    if fp:
        lines += [
            "",
            "### `--fix` projection",
            "",
            f"Would emit **{fp.get('emitted', 0)}** records: "
            f"{fp.get('unchanged', 0)} unchanged · {fp.get('fixed', 0)} fixed · "
            f"{fp.get('dropped', 0)} dropped · {fp.get('unparseable', 0)} unparseable",
        ]

    ts = s.token_stats
    if ts:
        lines += [
            "",
            "### Token budget",
            "",
            f"Budget **{ts.get('max_seq_len')}** · counting **{ts.get('counting')}** "
            f"({ts.get('method', '')})",
            "",
            "| over budget | at risk (≥85%) | median | max |",
            "|---:|---:|---:|---:|",
            f"| {ts.get('over_budget', 0)} | {ts.get('at_risk', 0)} "
            f"| {ts.get('median_tokens', 0)} | {ts.get('max_tokens', 0)} |",
        ]

    ds = s.duplicate_stats
    fb = s.format_breakdown
    lines += [
        "",
        "### Dataset shape",
        "",
        f"- duplicate clusters: **{ds.get('clusters', 0)}** "
        f"({ds.get('duplicate_records', 0)} duplicate records, "
        f"largest cluster {ds.get('largest_cluster', 0)})",
        f"- declared format: `{fb.get('declared_format', '?')}`",
        "- detected formats: " + (", ".join(
            f"`{k}` × {v}" for k, v in fb.get("detected_formats", {}).items()) or "—"),
        "- roles: " + (", ".join(
            f"`{k}` × {v}" for k, v in fb.get("roles", {}).items()) or "—"),
        f"- turns per record (min/median/max): "
        f"{fb.get('turns_per_record', {}).get('min', 0)} / "
        f"{fb.get('turns_per_record', {}).get('median', 0)} / "
        f"{fb.get('turns_per_record', {}).get('max', 0)}",
        f"- conversations ending on assistant: {fb.get('ends_on_assistant', 0)}/{n}",
        "",
    ]
    return "\n".join(lines)
