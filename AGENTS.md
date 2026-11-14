# AGENTS.md

Instructions for coding agents (Codex, Claude) working in this repository.

## What this is

Parallelogram is a strict, local-first CLI that validates LLM fine-tuning
datasets before a training run. The product promise is a single sentence:

> If `parallelogram check` exits 0 with all rules enabled, your training run
> won't fail because of data.

Every change is judged against that promise. A change that makes the CLI more
lenient, quieter about a real problem, or able to exit 0 on data that would
break training is a regression, even if tests pass.

## Layout

```
cli/                      the Python package (the product), published to PyPI
  src/parallelogram/
    cli.py                typer app: `check`, `report`
    formats/              on-disk format parsers; normalize to internal shape
    rules/                one module per rule id
    core/                 runner, fixer, summary/report, tokenization, io
    output/               terminal, JSON, markdown renderers
    optimize/             `optimize plan|run|report|export`: data selection
                          measured by real training (optional ML extras)
  tests/                  pytest suite (the source of truth)
  smoke.py, smoke_fix.py  stdlib-only smoke scripts
  examples/               small sample datasets used by tests and CI
landing/                  static site for parallelogram.dev (Vercel)
  index.md                the machine-readable page; source of the Agent view
scripts/                  release, CSP check and Agent view helpers
```

## Commands

Run from `cli/`:

```bash
python -m pip install -e '.[dev]'
pytest -q
python smoke.py && python smoke_fix.py
parallelogram check examples/clean.jsonl          # must exit 0
parallelogram check examples/broken.jsonl         # must exit 2
```

The optimize tests that train a model are skipped without the extras. To run
them: `python -m pip install -e '.[dev,optimize]'`, then `pytest -q`.

Every PR must leave `pytest -q` green. Add tests for every behavior you add or
change; put them in `cli/tests/`.

## Invariants (do not break)

- **Offline.** No network calls, no telemetry, no analytics, ever. Optional
  tokenizer extras may load local/HF tokenizers only when the user asks.
- **Runtime dependencies** are `typer` and `rich`. Do not add another without
  a stated reason in the PR description. Optional extras go in
  `[project.optional-dependencies]`.
- **Exit codes are a public API:** 0 clean, 1 warnings only (or partial fix),
  2 errors, 3 an operational failure (unreadable input or output, invalid
  `--eval` file) or a regression vs `--baseline` in `report`. Do not change
  them. MCP tool results carry the same `exit_code`.
- **Informational notes never affect the exit code.** A clean dataset must exit
  0 on a default install (no extras).
- **`schema` cannot be disabled.** Unknown rule ids passed to `--disable` are
  rejected, not ignored.
- **Normalize at the parse boundary.** Each format parser in `formats/` yields
  `ParseResult` records already converted to the internal OpenAI-chat shape
  `{"messages": [{"role", "content"}, ...]}` and tags `source_format`. Rules,
  the fixer and the report must stay format-agnostic.
- **Round-trip on output.** `check --fix --output` writes each record back in
  the format it was read in (see `to_sharegpt` and `_to_disk` in `cli.py`).
- **Never coerce silently.** Unknown or malformed input passes through to the
  `schema` rule so it is reported, not guessed at.
- **`check` and `report` must never disagree.** `report` is built from the
  same validation pass plus a dry fix pass.
- **`optimize` stays optional and honest.** torch, transformers, peft, numpy
  and yaml are imported only inside `optimize/` code paths that need them;
  importing `parallelogram`, the CLI or the MCP server must not load them.
  Models download only with `--allow-download`. Final test examples are read
  only after a freeze record exists, and never reach selection code. Reports
  keep estimated scores, development results and held-out results apart, and
  no test may depend on the model-aware selector beating a baseline.

## Adding a dataset format

1. Add `formats/<name>.py` with `is_<name>_record`, `normalize`,
   `iter_jsonl`, and an inverse `to_<name>` for `--fix` output.
2. Register it in `formats/__init__.py` (`get_parser`, `supported_formats`).
3. Teach `formats/auto.py` to detect it per line, without changing how
   existing shapes are detected.
4. Wire the inverse into `_to_disk` in `cli.py`.
5. Add an example file under `cli/examples/` and tests covering: parse,
   auto-detection, every rule firing on the new shape, and a `--fix --output`
   round trip.

## Scope discipline

- Do the task in the issue or prompt; do not refactor unrelated code.
- Do not bump the version, edit `CHANGELOG.md`, tag releases, or change the
  landing page unless the task says so. Releases are cut separately.
- Do not commit generated datasets, virtualenvs, `dist/`, or `.DS_Store`.
- Commit messages: `area: summary` (e.g. `cli: add --version flag`,
  `ci: run CLI tests on push`).

## Style

- Python 3.10+, type hints, `from __future__ import annotations`.
- Docstrings explain *why* a rule or behavior exists, not just what it does.
- CLI output: plain, precise, no emojis.
