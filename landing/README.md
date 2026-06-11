# parallelogram.dev — landing page

Static landing page for parallelogram. No build step. Open `index.html` in a browser.

## Files

- `index.html` — the page
- `styles.css` — the styles
- `demo.js` — the **interactive terminal demo**: an in-browser re-implementation
  of the linter that mirrors the real Python rules and CLI output
- `script.js` — motion (GSAP-driven reveals + IntersectionObserver fallback,
  hero parallax, magnetic buttons, scroll-spy, brand draw-on), install-chip
  copy, the compare demo, exit-code cycler, stat counters, cookie banner

## Dependencies

One CDN dependency, loaded `defer` and feature-detected: **GSAP + ScrollTrigger**
(`cdn.jsdelivr.net/npm/gsap`). The page is fully functional without it — if it
fails to load, or the visitor has `prefers-reduced-motion`, reveals fall back to
an IntersectionObserver and the rest of the motion is simply skipped. No build
step, no other deps.

## The interactive demo (`demo.js`)

The `#demo` terminal auto-types and lints a bundled sample on a loop **until the
visitor interacts** — clicking a sample chip (`broken.jsonl` / `clean.jsonl`),
editing the JSONL editor, or hitting Run hands control over; an "auto-demo"
control resumes the loop.

Its linter is a faithful port of the real rules (`cli/src/parallelogram/rules/*`)
and terminal output (`cli/src/parallelogram/output/terminal.py`), running in the
CLI's **approximate** token-counting mode (no tiktoken/HF in a browser). That
means it always emits the one-time "context-window counts are approximate" note
and treats overflow as a warning, exactly like `parallelogram check <file>` with
no `--tokenizer`. The bundled samples are the project's own fixtures
(`cli/examples/broken.jsonl`, `clean.jsonl`) so the demo can't drift from the CLI.
If you change a rule, update the matching `check*` function in `demo.js`.

## Design notes

**Aesthetic.** Refined-minimal, precision-instrument feel. Linear-style restraint with deeper monochrome. Single linter-green accent (the "exits 0" feeling); red used only inside demos for error states; amber only for warning markers.

**Type.** Geist (sans) + Geist Mono + Instrument Serif (italic, accent only). All Google Fonts, no Inter.

**The "video".** The macOS terminal panel runs the interactive demo (see above) — it auto-types a command, streams errors with line numbers and rule ids, and prints the summary panel, looping until the visitor takes over. It only starts when scrolled into view to save CPU.

**The mobile usecase.** The phone frame mocks a CI/PR-check view (GitHub-style notification card with the four issue rows from the demo). This previews the natural Phase 2 surface — CI integration — without overpromising what v0.1 ships.

**No tracking, no fonts beyond Google Fonts, no backend.** Drop the directory on any static host (Vercel, Netlify, Cloudflare Pages, GitHub Pages) and it works.

## Hosting

```
vercel deploy
# or
netlify deploy --dir=.
```

For `parallelogram.dev` specifically: any static host pointed at this folder, with the domain CNAME'd over.

## Tweak knobs

- All colors are CSS vars at the top of `styles.css` under `:root`.
- The terminal sequence is the `seq` array at the top of `script.js`; reorder, add, or change pacing freely.
- Section spacing lives in the `padding` of each `<section>`'s rule (`.hero`, `.demo`, `.rules`, etc.).
