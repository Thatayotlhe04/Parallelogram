# parallelogram.dev — landing page

Static landing page for parallelogram. No build step. Open `index.html` in a browser.

## Files

- `index.html` — the page
- `styles.css` — the styles
- `script.js` — the terminal animation, install-chip copy, reveal-on-scroll

## Design notes

**Aesthetic.** Refined-minimal, precision-instrument feel. Linear-style restraint with deeper monochrome. Single linter-green accent (the "exits 0" feeling); red used only inside demos for error states; amber only for warning markers.

**Type.** Geist (sans) + Geist Mono + Instrument Serif (italic, accent only). All Google Fonts, no Inter.

**The "video".** The macOS terminal panel runs an animated typing sequence on loop — types a command, streams errors with line numbers and rule ids, prints a summary, clears, retypes with `--output`, prints clean. It only starts when scrolled into view to save CPU.

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
