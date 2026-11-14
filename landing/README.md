# parallelogram.dev

A buildless, static landing page for the [Parallelogram CLI](https://github.com/Thatayotlhe04/Parallelogram), hosted on Vercel.

## Files and development

- `index.html`: hero, six checks, check/fix/gate workflow, native FAQ, closing install action.
- `styles.css`: shared layout and legal-page styles; local Manrope, Inter and Source Code Pro fonts (SIL OFL).
- `main.js`: install copy buttons only, with an accessible status announcement. With JavaScript or clipboard access unavailable, commands remain selectable.
- `privacy.html` and `terms.html`: legal pages with shared navigation and styling.
- The mark is the parallelogram ruled into a 3x3 lattice, white on black. `BRAND` in `scripts/build_shared.py` is the source for every header and footer; `favicon.svg`, `favicon-32x32.png`, `apple-touch-icon.png` and the mark on `og.png` use the same geometry.
- `og.png`: social image. Crawler files and `.well-known/security.txt` remain in place.
- `vercel.json`: unchanged first-party CSP and security headers.

Serve `landing/` with `python -m http.server 8000 --directory landing` from the repository root. Vercel uses `landing/` as its root, with no build command or output directory.

Run `python scripts/check_csp.py` from the repository root. Check desktop and 375px layouts, install buttons, FAQ keyboard operation, and legal-page links before publishing. No external resources, telemetry, cookies, storage, or runtime dataset demo.

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

## Why-check-first section

`#benchmarks` shows one measured result: training-data damage in a controlled
small-model test, and the defects the CLI can catch. `benchmarks.js` only
switches the chart between the 10% and 30% injection settings. The earlier API
cost calculator and ownership comparison were removed: they argued about model
ownership, not about the linter, and depended on a dated price snapshot.

The quality data is preserved in `benchmark-evidence.json`, with provenance
from research revision `1fcefb66637638ead2c50ab60d49ffaada57d769`. These are small-model
and validator experiments, not evidence of linter-induced accuracy gains.

Test the injection selector, 375px layout, reduced motion and no-JS fallback;
the selector and entrance have automated tests in `cli/tests/`.

Comparison motion is local CSS: a one-time viewport entrance, staggered column
reveals, smooth bar resizing, and hover highlights for columns and controls.
Hover decoration only runs on fine pointers; keyboard focus has equivalent
control emphasis. `prefers-reduced-motion` disables all comparison animations
and transitions. No looping effects, animation libraries or hidden initial content.

## Product walkthrough

The CLI panel is two windows on `scene-backdrop.webp` (an original, generated
painting). Behind: an editor showing lines 2128–2149 of the demo dataset's
`data/train.jsonl`, exactly as the file has them, with the pasted key on line
2131 selected. Its file, folder and branch icons are inline SVG, not symbol
glyphs, so they do not depend on the visitor's fonts. In front: `demo/terminal-108.mp4`, a screen recording of a real
bash session in that project. Below 760px only the 60-column recording
(`demo/terminal-60.mp4`) shows.

`scripts/demo/record.py` builds the synthetic 2,412-ticket support export
(`scripts/demo_dataset.py`) in a temporary git project, drives an interactive
bash in a real pty with scripted keystrokes, and saves an asciicast. Readline,
parallelogram and jq produce every byte on screen. `scripts/demo/render.py`
replays the cast through xterm.js in Chrome at 2x and encodes H.264 plus a
poster of the finished run.

Names and contacts in the dataset are fictional (Fabrikam/Contoso, Ofcom's
07700 900xxx drama range) and the API key is random characters.

`scripts/capture_terminal_demo.py` re-runs every recorded command against a
fresh copy of the project and fails if any output differs from the recording,
then keeps the page's transcript and editor lines in step. The video plays only while visible,
pauses on keyboard focus and in hidden tabs, and never plays under reduced
motion (the poster and transcript show instead). No playback controls.

From the repo root, with the local CLI installed in the current Python env:

```sh
python scripts/capture_terminal_demo.py            # verify the recording against the CLI
python scripts/capture_terminal_demo.py --record   # re-record both sessions (bash, git, jq)
npm install --prefix scripts/demo                  # xterm.js and JetBrains Mono for rendering
python scripts/demo/render.py scripts/demo/session-108.cast landing/demo/terminal-108.mp4 landing/demo/terminal-108.webp
python scripts/demo/render.py scripts/demo/session-60.cast landing/demo/terminal-60.mp4 landing/demo/terminal-60.webp
python scripts/capture_terminal_demo.py --write    # refresh the transcript block
```

Tests cover the recording against a fresh CLI run, the media files, visibility
and focus pausing, the narrow recording and reduced motion.
