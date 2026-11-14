"""Parallelogram Optimize: model-aware training-data selection, measured.

Given candidate data, a model, development examples, a recipe and a compute
budget, recommend a training subset and measure, by actually training,
whether it improved the quality/compute tradeoff against strong baselines.

    from parallelogram.optimize import load_experiment, plan, run, report, export

    experiment = load_experiment("experiment.yaml")
    plan(experiment)                    # the matrix and its cost; trains nothing
    run(experiment)                     # score, select, train, compare, freeze, test
    report("runs/my-run")               # what was estimated, measured, not done
    export("runs/my-run", "selected.jsonl")

Importing this package is cheap: torch, transformers and peft load only when
something has to be tokenized or trained (`pip install 'parallelogram[optimize]'`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import Experiment, check_model_refs, load_experiment, parse_experiment
from .errors import (ConfigError, LeakageError, MissingDependencyError, OptimizeError, OutOfMemory,
                     SelectionInfeasible, SplitIsolationError, UnsupportedError)
from .export import export
from .reporting import build_report, render_markdown

__all__ = [
    "ConfigError", "Experiment", "LeakageError", "MissingDependencyError", "OptimizeError", "OutOfMemory",
    "SelectionInfeasible", "SplitIsolationError", "UnsupportedError", "export", "load_experiment",
    "parse_experiment", "plan", "render_markdown", "report", "run",
]


def _experiment(experiment: Experiment | str | Path) -> Experiment:
    return experiment if isinstance(experiment, Experiment) else load_experiment(experiment)


def plan(experiment: Experiment | str | Path, *, allow_download: bool = False) -> dict:
    """The experiment matrix, resource assumptions, budget fit and expected
    artifacts. Reads data and counts tokens; trains nothing and writes nothing.

    Token counts are exact when the ML extras and the tokenizer are available
    locally, and labeled as estimates otherwise.
    """
    from .planner import build_plan, plan_to_dict, prepare

    exp = _experiment(experiment)
    check_model_refs(exp)
    info: dict = {}
    count = None
    try:
        from .backend import Backend
        backend = Backend(exp, allow_download=allow_download, tokenizer_only=True)
        count = backend.count_tokens
        info["library_versions"] = backend.versions
    except OptimizeError as exc:
        info["tokenizer_unavailable"] = str(exc)
    prep = prepare(exp, count)
    return plan_to_dict(prep, build_plan(prep), backend_info=info)


def run(experiment: Experiment | str | Path, *, allow_download: bool = False, final: bool = True,
        log: Callable[[str], None] | None = None) -> dict:
    """Run or resume an experiment and return its report.

    `final=False` stops after the development comparisons, leaving the run
    open for further search; the test split stays sealed.
    """
    from .runner import run as _run

    return _run(_experiment(experiment), allow_download=allow_download, final=final, log=log)


def report(run_dir: str | Path) -> dict:
    """The report for a run directory, rebuilt from its artifacts."""
    return build_report(run_dir)
