/* parallelogram.dev — hero console + data eater
   ────────────────────────────────────────────────────────────────────────
   The three-pane product surface at the top of the page: dirty.jsonl on
   the left, the live `parallelogram check` transcript in the middle,
   clean.jsonl on the right. This file owns the middle pane's typing and
   keeps the side panes in sync with it:

   - when a diagnostic prints, the matching dirty row (data-line="N")
     lights up in the left pane;
   - when the fix summary prints, the clean pane's rows materialize
     (.is-pending is lifted with a stagger).

   The transcript matches the real CLI's output format byte-for-byte
   (cli/src/parallelogram/output/terminal.py + the --fix summary in
   cli.py). Typing is deterministic: per-character delays come from a
   fixed table, never Math.random(), so every visit renders the same run.
   It plays once on scroll-in; reduced motion paints the finished state.

   Below the console, the eater strip: a small square agent travels the
   grid line and eats corrupted-data tokens (roles, dupe, mojibake, empty,
   JSON, >ctx), each collapsing into a clean status dot — bad data goes
   in, clean data comes out. Loops while visible; static under reduced
   motion; entirely aria-hidden (it is decoration, not content).

   NOTE: the mojibake glyph is written as \u escapes — the Windows-1252
   reading of the right single quote's UTF-8 bytes — same convention as
   demo.js, so the artifact survives any editor/encoding handling.
*/
(() => {
  'use strict';

  const console_ = document.getElementById('hero-console');
  const pane = document.getElementById('hero-check-body');
  const target = document.getElementById('hero-type-target');
  if (!console_ || !pane || !target) return;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const MOJIBAKE_APOS = 'â€™';

  const esc = s => String(s).replace(/[&<>]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function flagRow(n) {
    console_.querySelector(`[data-line="${n}"]`)?.classList.add('is-flagged');
  }

  function revealClean(stagger) {
    const rows = Array.from(console_.querySelectorAll('#hero-pane-clean .is-pending'));
    rows.forEach((row, i) => {
      if (stagger) setTimeout(() => row.classList.remove('is-pending'), i * 110);
      else row.classList.remove('is-pending');
    });
  }

  // ── the transcript ─────────────────────────────────────────────────
  const CMD =
    'parallelogram check train.jsonl --tokenizer gpt-4o --fix --output clean.jsonl';

  // [pause-before-ms, css-class, html, side-effect?]
  const LINES = [
    [340, 'blank', '&nbsp;'],
    [120, '',
      `  <span class="err">✗</span> <span class="lno">train.jsonl:42</span> ` +
      `<span class="rid">[roles]</span> Conversation must end on 'assistant', ended on 'user'`,
      () => flagRow(42)],
    [90, 'detail',
      `      <span class="dim">→ unfixable — record will be dropped</span>`],
    [320, '',
      `  <span class="warn">!</span> <span class="lno">train.jsonl:97</span> ` +
      `<span class="rid">[encoding]</span> Message 1 contains likely mojibake: '${esc(MOJIBAKE_APOS)}'`,
      () => flagRow(97)],
    [90, 'detail',
      `      <span class="dim">→ fixed — UTF-8 → latin-1 → UTF-8 round-trip artifact</span>`],
    [320, '',
      `  <span class="err">✗</span> <span class="lno">train.jsonl:131</span> ` +
      `<span class="rid">[context-window]</span> Record exceeds max_seq_len: 5234 &gt; 4096 tokens`,
      () => flagRow(131)],
    [90, 'detail',
      `      <span class="dim">→ counted with tiktoken o200k_base (gpt-4o) — truncated to fit</span>`],
    [380, 'blank', '&nbsp;'],
    [110, '', `  <span class="ok">✓</span> <span class="rid">encoding</span> · 1 fix`],
    [90, '', `  <span class="ok">✓</span> <span class="rid">context-window</span> · 1 fix`],
    [320, 'blank', '&nbsp;'],
    [110, '', `  <span class="err">✗</span> dropped:`],
    [90, 'detail',
      `      <span class="dim"><span class="lno">train.jsonl:42</span> → <span class="rid">roles</span> (unfixable)</span>`],
    [380, 'blank', '&nbsp;'],
    [120, 'panel',
      `<span class="t-panel border-amber"><span class="t-panel-title">parallelogram</span>` +
      `<span class="sum-strong">547 records</span>  ` +
      `<span class="dim">544 unchanged</span>  ` +
      `<span class="ok">2 fixed</span>  ` +
      `<span class="err">1 dropped</span>  ` +
      `<span class="dim">0 unparseable</span></span>`],
    [240, '',
      `  <span class="ok">→ clean.jsonl written · 546 records · exit 1</span>`,
      () => revealClean(true)],
  ];

  // Fixed per-character delay table — cycled, so the cadence reads human
  // but is identical on every visit.
  const DELAYS = [42, 34, 58, 38, 46, 30, 62, 40, 36, 50];

  function appendLine(cls, html) {
    const el = document.createElement('span');
    el.className = `t-line ${cls}`;
    el.innerHTML = html;
    pane.appendChild(el);
    pane.scrollTop = pane.scrollHeight;
  }

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const whileVisible = async () => {
    while (document.hidden) await sleep(250);
  };

  function paintFinal() {
    target.insertAdjacentText('beforeend', CMD);
    for (const [, cls, html, fx] of LINES) {
      appendLine(cls, html);
      if (fx) fx();
    }
    revealClean(false);
    pane.scrollTop = 0; // show the command first; the pane scrolls
  }

  async function run() {
    if (reduceMotion) {
      paintFinal();
      return;
    }
    const cur = document.createElement('span');
    cur.className = 'cur';
    target.appendChild(cur);

    for (let i = 0; i < CMD.length; i++) {
      await whileVisible();
      cur.insertAdjacentText('beforebegin', CMD[i]);
      await sleep(DELAYS[i % DELAYS.length]);
    }
    await sleep(180);
    cur.remove();

    for (const [pause, cls, html, fx] of LINES) {
      await whileVisible();
      await sleep(pause);
      appendLine(cls, html);
      if (fx) fx();
    }
  }

  let started = false;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !started) {
        started = true;
        run();
        io.disconnect();
      }
    }
  }, { threshold: 0.2 });
  io.observe(console_);
})();

/* ── the data eater ──────────────────────────────────────────────────── */
/* A pixel-art pac-man marching the open ground above the install row,
   eating corrupted-data tokens into clean status dots. It runs edge to
   edge of the row below; at the right corner it disappears and reappears
   at the left, continuously. Two pixel frames (open/closed mouth) are
   flipped by CSS steps() — no tweening, it's a flip-book. Static under
   reduced motion; aria-hidden throughout (decoration, not content). */
(() => {
  'use strict';

  const strip = document.getElementById('eater-strip');
  const agent = document.getElementById('eater-agent');
  if (!strip || !agent) return;

  // 12×12 pixel maps, right-facing. '#' = lit pixel.
  const PIX_OPEN = [
    '....####....',
    '..########..',
    '.##########.',
    '.######.....',
    '######......',
    '#####.......',
    '#####.......',
    '######......',
    '.######.....',
    '.##########.',
    '..########..',
    '....####....',
  ];
  const PIX_CLOSED = [
    '....####....',
    '..########..',
    '.##########.',
    '.##########.',
    '############',
    '############',
    '############',
    '############',
    '.##########.',
    '.##########.',
    '..########..',
    '....####....',
  ];

  function frame(map, cls) {
    const rects = [];
    map.forEach((row, y) => {
      // contiguous runs become single rects — fewer nodes, same pixels
      let x = 0;
      while (x < row.length) {
        if (row[x] !== '#') { x++; continue; }
        let w = 0;
        while (row[x + w] === '#') w++;
        rects.push(`<rect x="${x}" y="${y}" width="${w}" height="1"/>`);
        x += w;
      }
    });
    return `<g class="${cls}" fill="currentColor">${rects.join('')}</g>`;
  }

  agent.innerHTML =
    `<svg viewBox="0 0 12 12" aria-hidden="true">` +
    frame(PIX_OPEN, 'frame-open') + frame(PIX_CLOSED, 'frame-closed') +
    `</svg>`;

  const tokens = Array.from(strip.querySelectorAll('.eater-token'));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reduceMotion) {
    // Static frame: agent mid-track, everything behind it already eaten.
    agent.style.left = '60%';
    tokens.forEach((t) => {
      if (parseFloat(t.style.getPropertyValue('--x')) < 60) t.classList.add('eaten');
    });
    return;
  }

  const TRAVEL_MS = 6000;   // one full left → right pass
  const X_START = -4;       // off the left corner …
  const X_END = 104;        // … to past the right corner (then wrap)
  let rafId = 0;
  let visible = false;
  let lastX = X_START;

  function frameTick(now) {
    if (!visible) return;
    const t = (now % TRAVEL_MS) / TRAVEL_MS;
    const x = X_START + t * (X_END - X_START);

    // wrapped: it left at the right corner, it re-enters at the left —
    // the eaten tokens respawn for the next pass
    if (x < lastX) tokens.forEach(tok => tok.classList.remove('eaten'));
    lastX = x;

    agent.style.left = x + '%';
    for (const tok of tokens) {
      if (!tok.classList.contains('eaten')
          && x >= parseFloat(tok.style.getPropertyValue('--x'))) {
        tok.classList.add('eaten');
      }
    }
    rafId = requestAnimationFrame(frameTick);
  }

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      visible = e.isIntersecting && !document.hidden;
      strip.classList.toggle('is-running', visible);
      cancelAnimationFrame(rafId);
      if (visible) rafId = requestAnimationFrame(frameTick);
    }
  }, { threshold: 0.3 });
  io.observe(strip);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      visible = false;
      strip.classList.remove('is-running');
      cancelAnimationFrame(rafId);
    }
  });
})();
