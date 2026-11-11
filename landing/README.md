# parallelogram.dev

A buildless, static landing page for the [Parallelogram CLI](https://github.com/Thatayotlhe04/Parallelogram), hosted on Vercel.

## Files and development

- `index.html`: hero, six checks, check/fix/gate workflow, native FAQ, closing install action.
- `styles.css`: shared layout and legal-page styles; local Manrope, Inter and Source Code Pro fonts (SIL OFL).
- `main.js`: install copy buttons only, with an accessible status announcement. With JavaScript or clipboard access unavailable, commands remain selectable.
- `privacy.html` and `terms.html`: legal pages with shared navigation and styling.
- `og.png`: existing social image. Favicons, crawler files, and `.well-known/security.txt` remain in place.
- `vercel.json`: unchanged first-party CSP and security headers.

Serve `landing/` with `python -m http.server 8000 --directory landing` from the repository root. Vercel uses `landing/` as its root, with no build command or output directory.

Run `python scripts/check_csp.py` from the repository root. Check desktop and 375px layouts, install buttons, FAQ keyboard operation, and legal-page links before publishing. No external resources, telemetry, cookies, storage, animation, or runtime dataset demo.

## Terminal provenance

The static transcript was captured by running the checked-out CLI at `d99eed3` (package version 0.4.2), with all six rules enabled and no optional tokenizer. The page targets the concurrent v1.0 release described in the handoff brief. Before merging, review release-dependent claims against that release.

The four input records are: a valid user/assistant exchange; an exact copy; a valid exchange preceded by a whitespace-only system message; and an exchange with a BOM inside the user message. The check emits three issues: duplicate, empty content, and encoding. Fixing removes the duplicate and repairs the other two records, so fix exits **1** for the dropped record; the explicit clean-file re-check exits **0**.

Reproduce in a scratch directory with the CLI installed:

```python
import json
from pathlib import Path

records = [
    {"messages": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]},
    {"messages": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]},
    {"messages": [{"role": "system", "content": " "}, {"role": "user", "content": "What is a gradient?"}, {"role": "assistant", "content": "A vector of partial derivatives."}]},
    {"messages": [{"role": "user", "content": "\ufeffWhat is a tensor?"}, {"role": "assistant", "content": "A multidimensional array."}]},
]
Path("train.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
```

```sh
parallelogram check train.jsonl                         # exit 2
parallelogram check train.jsonl --fix --output clean.jsonl # exit 1
parallelogram check clean.jsonl                         # exit 0
```

The display removes Rich box borders, blank lines, indentation, and two explanatory detail lines. Diagnostic text, summary counts, and command order are unchanged; `exit N` annotations show captured process statuses, not CLI-generated lines. The approximate-token-count note is retained.

## Model comparison section

`#benchmarks` uses vertical, provider-colored columns and local brand marks.
Its three views are cost at scale, ownership/control tradeoffs and measured
training-data quality. There is no paper promotion or model intelligence score.

`benchmarks.js` holds a dated official-price snapshot (12 November 2026), two
API tiers and a browser-local scenario calculator. Costs assume 1,500 input and
200 total billed output tokens per task. The owned deployment is an editable
budget, not a measured model cost. The default includes $2,800 hosting, $2,500
maintenance and $700 amortized training. Capacity is assumed, not simulated.
Gemini 4 Carbon's promotional rates expire November 30; update the snapshot
and source text together if refreshing prices. No runtime network requests.

The quality data is preserved in `benchmark-evidence.json`, with provenance
from research revision `1fcefb66637638ead2c50ab60d49ffaada57d769`. These are small-model
and validator experiments, not evidence of linter-induced accuracy gains.

Brand SVGs in `logos/` come from `@lobehub/icons-static-svg` 1.102.0 (MIT; license
included). Logos identify providers and model families; no endorsement implied.

Test the three tabs, arrows/Home/End, API tiers, volume buttons, invalid budgets,
quality selector, 375px layout, reduced motion and no-JS fallback. Both all-panel
fallback and the scenario arithmetic have automated tests in `cli/tests/`.
