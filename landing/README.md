# parallelogram.dev — landing site

Static landing page for the [parallelogram](https://github.com/Thatayotlhe04/Parallelogram)
CLI — preflight validation for fine-tuning datasets. No build step, no framework,
no third-party requests: serve the folder and it works.

## Files

| File | Purpose |
|---|---|
| `index.html` | The landing page — 14 numbered sections (see the TOC comment at the top of the file) |
| `styles.css` | All styling, organized with CSS cascade layers: `tokens, base, components, sections, legal, motion, utilities` |
| `main.js` | Header state, mobile nav, reveals, scroll-spy, copy buttons, token meter, compare stream, architecture inspector, exit-code cycler, cookie banner |
| `hero-terminal.js` | The scripted hero terminal — deterministic typing (fixed delay table, no RNG), three tabs |
| `demo.js` | In-browser port of the six linter rules plus the `--fix` pipeline and ShareGPT normalizer, faithful to `cli/src/parallelogram/`; exposes `window.__pgLint` / `window.__pgFix` for parity testing |
| `privacy.html` / `terms.html` | Legal pages, styled by the `legal` CSS layer |
| `fonts/` | Self-hosted Geist + Geist Mono variable woff2 (OFL) — no Google Fonts |
| `og.png` | 1200×630 social preview (`og-image.png` is the legacy image, kept so previously shared links don't break) |
| `vercel.json` | Immutable cache headers for `/fonts/*` |

## Local development

```bash
cd landing
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy (Vercel)

Static deployment: set the project root directory to `landing/`, no build
command, no output directory. Any other static host works identically.

## Conventions

- **No external resources.** No GSAP, no CDN scripts, no Google Fonts —
  everything is first-party (the privacy policy now promises exactly this).
- **Legacy anchors** `#rules`, `#install`, and `#integrations` are kept as
  zero-height alias anchors so old inbound links and the legal pages keep working.
- **Reduced motion** is fully supported: typing paints its final frame,
  streams render one static pass, reveals are instant.
- **Mojibake strings in `demo.js` are written as `\u` escapes**, never literal
  glyphs, so rule matching survives any editor/encoding round-trip.
- The demo must stay in lockstep with the CLI: if a rule changes in
  `cli/src/parallelogram/rules/`, update the matching `check*`/`fix*` function
  in `demo.js`.

## Tweak knobs

- Design tokens (colors, spacing, type, easing) live in `:root` inside the
  `tokens` layer at the top of `styles.css`.
- The hero transcript is the `LINES` array in `hero-terminal.js`.
- Demo fixtures are the `SAMPLES` object in `demo.js` (per format).
