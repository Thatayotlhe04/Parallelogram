/* parallelogram.dev — hero terminal
   ────────────────────────────────────────────────────────────────────────
   The scripted demo at the top of the page: three tabs (dirty.jsonl /
   parallelogram check / clean.jsonl). The dirty and clean panes are static
   HTML; this file owns the check pane, typing the command and streaming a
   transcript that matches the real CLI's output format byte-for-byte
   (cli/src/parallelogram/output/terminal.py + the --fix summary in cli.py).

   The animation is deterministic: per-character delays come from a fixed
   table, never Math.random(), so every visit renders the identical run.
   It plays once on scroll-in (the live demo further down is the one that
   loops), pauses while the tab is hidden, and paints the finished
   transcript instantly under prefers-reduced-motion.

   NOTE: the mojibake glyph is written as \u escapes — the Windows-1252
   reading of the right single quote's UTF-8 bytes — same convention as
   demo.js, so the artifact survives any editor/encoding handling.
*/
(() => {
  'use strict';

  const root = document.getElementById('hero-term');
  const target = document.getElementById('hero-type-target');
  if (!root || !target) return;

  const pane = target.parentElement; // #hero-pane-check
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const MOJIBAKE_APOS = 'â€™';

  // ── tabs ───────────────────────────────────────────────────────────
  const tabs = Array.from(root.querySelectorAll('[data-hero-tab]'));
  const paneOf = tab => document.getElementById(tab.getAttribute('aria-controls'));

  function selectTab(tab) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      paneOf(t)?.toggleAttribute('hidden', !on);
    });
    tab.focus({ preventScroll: true });
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const next = e.key === 'ArrowRight' ? i + 1 : i - 1;
      selectTab(tabs[(next + tabs.length) % tabs.length]);
    });
  });

  // ── the transcript ─────────────────────────────────────────────────
  const CMD =
    'parallelogram check train.jsonl --tokenizer gpt-4o --fix --output clean.jsonl';

  const esc = s => String(s).replace(/[&<>]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  // [pause-before-ms, css-class, html]
  const LINES = [
    [340, 'blank', '&nbsp;'],
    [120, '',
      `  <span class="err">✗</span> <span class="lno">train.jsonl:42</span> ` +
      `<span class="rid">[roles]</span> Conversation must end on 'assistant', ended on 'user'`],
    [90, 'detail',
      `      <span class="dim">→ unfixable — record will be dropped</span>`],
    [320, '',
      `  <span class="warn">!</span> <span class="lno">train.jsonl:97</span> ` +
      `<span class="rid">[encoding]</span> Message 1 contains likely mojibake: '${esc(MOJIBAKE_APOS)}'`],
    [90, 'detail',
      `      <span class="dim">→ fixed — UTF-8 → latin-1 → UTF-8 round-trip artifact</span>`],
    [320, '',
      `  <span class="err">✗</span> <span class="lno">train.jsonl:131</span> ` +
      `<span class="rid">[context-window]</span> Record exceeds max_seq_len: 5234 &gt; 4096 tokens`],
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
      `  <span class="ok">→ clean.jsonl written · 546 records · exit 1</span>`],
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

  // ── the run (once; cancellable; hidden-tab aware) ──────────────────
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const whileVisible = async () => {
    while (document.hidden) await sleep(250);
  };

  function paintFinal() {
    target.insertAdjacentText('beforeend', CMD);
    for (const [, cls, html] of LINES) appendLine(cls, html);
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

    for (const [pause, cls, html] of LINES) {
      await whileVisible();
      await sleep(pause);
      appendLine(cls, html);
    }
  }

  // ── start on scroll-in, once ───────────────────────────────────────
  let started = false;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !started) {
        started = true;
        run();
        io.disconnect();
      }
    }
  }, { threshold: 0.25 });
  io.observe(root);
})();
