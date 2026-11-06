/* parallelogram.dev — copy buttons, CLI/CI tabs, and the eater.
   The page reads fine without JavaScript; everything here is enhancement. */
(() => {
  'use strict';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── Copy buttons ──────────────────────────────────────────────────
  const status = document.getElementById('copy-status');
  document.querySelectorAll('[data-copy]').forEach(button => {
    let reset;
    button.addEventListener('click', async () => {
      clearTimeout(reset);
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.classList.add('copied');
        button.querySelector('.copy-label').textContent = 'Copied';
        if (status) status.textContent = 'Copied';
      } catch {
        if (status) status.textContent = 'Could not copy. Select the command and copy it manually.';
      }
      reset = setTimeout(() => {
        button.classList.remove('copied');
        if (status) status.textContent = '';
      }, 2000);
    });
  });

  // ── CLI / CI tabs ─────────────────────────────────────────────────
  const tabs = Array.from(document.querySelectorAll('.pill-toggle [role="tab"]'));
  function select(tab) {
    tabs.forEach(t => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', e => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      select(next);
      next.focus();
    });
  });

  // ── The eater ─────────────────────────────────────────────────────
  // A pixel parallelogram walks the track above the install row and
  // chomps corrupted-data tokens into clean dots, then wraps around.
  // Two frames (open/closed jaw) are flipped by CSS steps().
  const strip = document.getElementById('eater');
  const agent = document.getElementById('eater-agent');
  if (!strip || !agent) return;

  // 17×14 pixel maps, facing right. '#' = lit. Open jaw / closed jaw.
  const OPEN = [
    '....############.',
    '....############.',
    '....#######..###.',
    '...########..##..',
    '...############..',
    '...#########.....',
    '..######.........',
    '..#########......',
    '..############...',
    '.############....',
    '.############....',
    '.############....',
    '############.....',
    '..##....##.......',
  ];
  const CLOSED = OPEN.map((row, r) => r >= 5 && r <= 7
    ? row.slice(0, row.indexOf('#')) + '#'.repeat(12) + '.'.repeat(17 - row.indexOf('#') - 12)
    : row);
  function frame(map, cls) {
    const rects = [];
    map.forEach((row, y) => {
      for (let x = 0; x < row.length;) {
        if (row[x] !== '#') { x++; continue; }
        let w = 0;
        while (row[x + w] === '#') w++;
        rects.push(`<rect x="${x}" y="${y}" width="${w}" height="1"/>`);
        x += w;
      }
    });
    return `<g class="${cls}" fill="currentColor">${rects.join('')}</g>`;
  }
  agent.innerHTML = '<svg viewBox="0 0 17 14" aria-hidden="true">' +
    frame(OPEN, 'frame-open') + frame(CLOSED, 'frame-closed') + '</svg>';

  const tokens = Array.from(strip.querySelectorAll('.eater-token'));
  const xOf = t => parseFloat(t.style.getPropertyValue('--x'));

  if (reduceMotion) {
    agent.style.left = '50%';
    tokens.forEach(t => { if (xOf(t) < 50) t.classList.add('eaten'); });
    return;
  }

  const TRAVEL_MS = 6500, X0 = -4, X1 = 104;
  let raf = 0, visible = false, lastX = X0;
  function tick(now) {
    if (!visible) return;
    const x = X0 + ((now % TRAVEL_MS) / TRAVEL_MS) * (X1 - X0);
    if (x < lastX) tokens.forEach(t => t.classList.remove('eaten'));
    lastX = x;
    agent.style.left = x + '%';
    for (const t of tokens) if (x >= xOf(t)) t.classList.add('eaten');
    raf = requestAnimationFrame(tick);
  }
  function run(on) {
    visible = on;
    strip.classList.toggle('is-running', on);
    cancelAnimationFrame(raf);
    if (on) raf = requestAnimationFrame(tick);
  }
  new IntersectionObserver(entries => {
    for (const e of entries) run(e.isIntersecting && !document.hidden);
  }, { threshold: 0.3 }).observe(strip);
  document.addEventListener('visibilitychange', () => { if (document.hidden) run(false); });
})();
