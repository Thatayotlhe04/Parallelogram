# parallelogram

The linter for fine-tuning data. If `parallelogram check` exits 0 with all rules enabled, your training run won't fail because of data.

Two parts in this bundle:

```
.
├── cli/        — the Python CLI (the actual product)
└── landing/    — the parallelogram.dev marketing page
```

## quickstart

Install from PyPI:

```bash
pip install parallelogram
parallelogram check examples/broken.jsonl
parallelogram check examples/broken.jsonl --fix --output clean.jsonl
```

Or develop locally:

```bash
cd cli
pip install -e .
parallelogram check examples/broken.jsonl
parallelogram check examples/broken.jsonl --fix --output clean.jsonl
```

To verify the engine without installing anything:

```bash
cd cli
python smoke.py        # 32 stdlib-only checks (Phase 1 — all rules)
python smoke_fix.py    # 22 stdlib-only checks (Phase 2 — fix tier)
```

See `cli/README.md` for full docs and exit-code semantics.

## cli/ — the validator

A strict pre-flight validator for fine-tuning datasets. Six rules covering the failure modes that silently corrupt training: schema, role sequences, empty turns, context-window overflow, exact duplicates, encoding artifacts.

In v0.2, `--fix` adds free local mechanical repair — strip BOM markers, replace mojibake, drop empty turns, truncate context-window overflow, deduplicate. Anything that still errors after the fix attempt is dropped from the output. SLM-tier fixes (rewriting broken role sequences, filling in incomplete assistant turns) are scoped but deferred — the architecture is in place, the implementation lands when there's user traction to inform what to build.

Open source under Apache 2.0. No telemetry. No backend. Pure local.

## landing/ — the marketing page

Static HTML/CSS/JS for `parallelogram.dev`. No build step.

```bash
cd landing
# open index.html in any browser, or:
python -m http.server 8000
```

Drop the `landing/` folder on Vercel, Netlify, Cloudflare Pages, or GitHub Pages and point `parallelogram.dev` at it.

See `landing/README.md` for design notes.

## status

**CLI v0.2.0** — Phase 1 (six validation rules) and Phase 2 step 1 (mechanical `--fix`) shipped and tested. Published to PyPI. 109 checks across smoke and end-to-end suites, all passing.

**Landing page v0.2** — quickstart section shows `--fix`, hero includes one-click pip install copy, cookie consent banner for privacy compliance, terms-of-use scopes warranty to v0.2 rules.

## roadmap

- ShareGPT and raw-completion format coverage
- Opt-in anonymized error-type analytics — informs whether the SLM-fix tier is worth building
- `--fix` SLM tier (paid, hosted) — gated on traction data from analytics phase
- Additional validation rules based on user feedback

Licensed under Apache 2.0.
