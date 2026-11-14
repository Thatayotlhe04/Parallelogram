"""`parallelogram optimize` commands.

    parallelogram optimize plan experiment.yaml
    parallelogram optimize run experiment.yaml
    parallelogram optimize report RUN_DIRECTORY
    parallelogram optimize export RUN_DIRECTORY --output selected.jsonl

Exit codes follow the rest of the CLI: 0 the command did what was asked,
2 the contract or the arguments are invalid (including a plan that does not
fit its budget), 3 an operational failure (missing extras, a model that is
not available locally, out of memory, unreadable files). A completed run
exits 0 whatever it concluded: "insufficient evidence" is a result, and it is
in the report, not in the exit code.

Only typer is imported at module level so that registering these commands
adds nothing to the start-up of `parallelogram check`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

optimize_app = typer.Typer(
    name="optimize",
    help="Recommend a training subset and measure, by training, whether it helps. "
         "Needs the optional extras: pip install 'parallelogram\\[optimize]'.",
    add_completion=False,
    no_args_is_help=True,
)


def _guard(action):
    """Run `action`, mapping optimize failures onto the CLI's exit codes."""
    from ..api import ParallelogramError
    from ..core.report import scrub

    try:
        return action()
    except ValueError as exc:  # ConfigError and its subclasses
        typer.echo(scrub(str(exc)), err=True)
        raise typer.Exit(2)
    except ParallelogramError as exc:  # OptimizeError and its subclasses
        typer.echo(scrub(str(exc)), err=True)
        raise typer.Exit(3)


def _emit_json(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _render_plan(plan: dict) -> None:
    echo = typer.echo
    data, budget = plan["data"], plan["budget"]
    echo(f"plan: {plan['name']}  ({plan['scale']}, {plan['objective']})")
    echo(f"  candidates   {data['candidates']['records']} records; pool {data['pool']['records']} records, "
         f"{data['pool']['tokens']} tokens "
         f"(excluded: {data['pool']['excluded_for_leakage']} leakage, {data['pool']['excluded_for_length']} length)")
    echo(f"  development  {data['dev']['selection_records']} examples steer selection, "
         f"{data['dev']['comparison_records']} compare policies"
         + ("  (the same examples: comparison favors the gradient selector)" if data["dev"]["shared_halves"] else ""))
    echo(f"  test         {data['test']['records']} examples, sealed until the candidate is frozen")
    echo(f"  token counts {data['token_counts']}")
    leak = plan["leakage"]
    echo(f"  split overlap {leak['exact_findings']} exact, {leak['uncertain_findings']} uncertain finding(s)")
    echo("")
    echo("experiment matrix")
    for row in plan["matrix"]:
        size = "all candidates" if row["token_budget"] is None else f"{row['token_budget']} tokens"
        echo(f"  {row['key']:<34} {row['comparison']:<16} {size:<18} seeds {row['seeds']}")
    echo("")
    echo(f"budget ({budget['unit']})")
    for stage, value in budget["estimated_by_stage"].items():
        echo(f"  {stage:<18} {value:>16,.0f}")
    echo(f"  {'estimated total':<18} {budget['estimated_total']:>16,.0f}")
    echo(f"  {'budget':<18} {budget['total_compute']:>16,.0f}")
    if plan["not_planned"]:
        echo("")
        echo("not planned")
        for item in plan["not_planned"]:
            echo(f"  {item['what']}: {item['reason']}")
    assumptions = plan["resource_assumptions"]
    echo("")
    echo(f"resources: device {assumptions['device']}, dtype {assumptions['dtype']}, max_seq_len "
         f"{assumptions['max_seq_len']}, batch {assumptions['batch_size']}, {assumptions['epochs']} epoch(s); "
         f"model downloads {assumptions['model_downloads']}")
    if assumptions.get("feature_storage_bytes"):
        echo(f"  gradient feature cache about {assumptions['feature_storage_bytes'] / 1e6:,.1f} MB")
    echo("")
    echo("expected artifacts")
    for path in plan["expected_artifacts"]:
        echo(f"  {path}")
    echo("")
    if plan["fits_budget"]:
        echo("Nothing was trained. Run it with: parallelogram optimize run <experiment>")
    else:
        for problem in plan["problems"]:
            echo(problem, err=True)


@optimize_app.command("plan")
def plan_command(
    experiment: Path = typer.Argument(..., help="Experiment contract (.yaml, .yml or .json)."),
    json_output: bool = typer.Option(False, "--json", help="Emit the plan as JSON."),
    allow_download: bool = typer.Option(False, "--allow-download",
                                        help="Allow fetching the tokenizer from the Hub."),
) -> None:
    """Show the experiment matrix, costs, budget fit and artifacts. Trains nothing."""
    from . import plan

    result = _guard(lambda: plan(experiment, allow_download=allow_download))
    if json_output:
        _emit_json(result)
    else:
        _render_plan(result)
    raise typer.Exit(0 if result["fits_budget"] else 2)


@optimize_app.command("run")
def run_command(
    experiment: Path = typer.Argument(..., help="Experiment contract (.yaml, .yml or .json)."),
    allow_download: bool = typer.Option(
        False, "--allow-download",
        help="Allow downloading the model and tokenizer from the Hub. Off by default: "
             "without it nothing leaves or enters this machine."),
    no_final: bool = typer.Option(
        False, "--no-final",
        help="Stop after the development comparisons. The run stays open for more "
             "search and the test split stays sealed."),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    quiet: bool = typer.Option(False, "--quiet", help="No progress lines on stderr."),
) -> None:
    """Score, select, train, compare, freeze and test. Resumes if interrupted."""
    from . import render_markdown, run

    log = None if quiet else (lambda message: typer.echo(f"[optimize] {message}", err=True))
    report = _guard(lambda: run(experiment, allow_download=allow_download, final=not no_final, log=log))
    if json_output:
        _emit_json(report)
    else:
        sys.stdout.write(render_markdown(report))
        typer.echo(f"report written to {report['run_directory']}/report.md", err=True)
    raise typer.Exit(0)


@optimize_app.command("report")
def report_command(
    run_directory: Path = typer.Argument(..., help="A run directory written by `optimize run`."),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Optional[Path] = typer.Option(None, "--out", help="Also write the report to this file."),
) -> None:
    """Report estimated signals, measured results, compute, coverage and limits."""
    from . import render_markdown, report
    from .cache import atomic_write

    result = _guard(lambda: report(run_directory))
    rendered = (json.dumps(result, indent=2, ensure_ascii=False) + "\n") if json_output else render_markdown(result)
    sys.stdout.write(rendered)
    if out is not None:
        _guard(lambda: _write_text(atomic_write, out, rendered))
        typer.echo(f"report written to {out}", err=True)
    raise typer.Exit(0)


def _write_text(atomic_write, path: Path, text: str) -> None:
    from .errors import OptimizeError

    try:
        atomic_write(path, lambda f: f.write(text), mode="w")
    except OSError as exc:
        raise OptimizeError(f"Could not write {path}: {exc}") from exc


@optimize_app.command("export")
def export_command(
    run_directory: Path = typer.Argument(..., help="A run directory written by `optimize run`."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the selected records."),
    trial: Optional[str] = typer.Option(None, "--trial", help="Export this arm instead of the frozen candidate."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Which seed's subset, for arms that differ by seed."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output file."),
    json_output: bool = typer.Option(False, "--json", help="Emit the manifest as JSON."),
) -> None:
    """Copy the selected records, unchanged, to a new file with a manifest."""
    from . import export
    from .errors import OptimizeError

    def action():
        try:
            return export(run_directory, output, trial=trial, seed=seed, overwrite=overwrite)
        except OSError as exc:
            raise OptimizeError(f"Could not export to {output}: {exc}") from exc

    manifest = _guard(action)
    if json_output:
        _emit_json(manifest)
    else:
        selection, evidence = manifest["selection"], manifest["evidence"]
        typer.echo(f"wrote {manifest['output']['records']} records ({manifest['output']['tokens']} tokens) to {output}")
        typer.echo(f"manifest: {output}.manifest.json")
        typer.echo(f"arm: {selection['trial']} (policy {selection['policy']})"
                   + ("" if selection["frozen_candidate"] else "  [not the frozen candidate]"))
        typer.echo(f"evidence: {evidence['conclusion']['status'].replace('_', ' ')} ({evidence['label']})")
    raise typer.Exit(0)
