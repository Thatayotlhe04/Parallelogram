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

  // ── Star count ────────────────────────────────────────────────────
  // The number is written into the page by .github/workflows/stars.yml, so
  // the site makes no third-party requests. It counts up once on load.
  document.querySelectorAll('.stars[data-stars]').forEach(el => {
    const target = Number(el.dataset.stars) || 0;
    const fmt = n => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);
    el.setAttribute('aria-label', `${target} GitHub stars`);
    if (reduceMotion || target === 0) { el.textContent = fmt(target); return; }
    const t0 = performance.now(), dur = 1200;
    const step = now => {
      const k = Math.min(1, (now - t0) / dur), eased = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(Math.round(target * eased));
      if (k < 1) requestAnimationFrame(step);
    };
    el.textContent = '0';
    requestAnimationFrame(step);
  });

  // ── Check / fix / gate: draws in as you scroll ────────────────────
  const diagram = document.querySelector('.diagram');
  if (diagram && !reduceMotion) {
    let ticking = false;
    const update = () => {
      ticking = false;
      const r = diagram.getBoundingClientRect(), vh = window.innerHeight;
      // 0 when the diagram's top reaches 90% of the viewport, 1 at 35%.
      const p = Math.min(1, Math.max(0, (0.9 * vh - r.top) / (0.55 * vh)));
      diagram.style.setProperty('--p', p.toFixed(3));
    };
    const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
  }

  // ── Works-with strip ──────────────────────────────────────────────
  // Logos are official files dropped into landing/logos/. A logo that
  // fails to load is hidden; the strip appears once three or more load,
  // so the page never shows a half-empty row.
  const works = document.getElementById('works');
  if (works) {
    const imgs = Array.from(works.querySelectorAll('img'));
    let settled = 0, loaded = 0;
    const done = (img, ok) => {
      settled++;
      if (ok) loaded++; else img.closest('li').classList.add('missing');
      if (settled === imgs.length && loaded >= 3) works.hidden = false;
    };
    imgs.forEach(img => {
      img.loading = 'eager';  // hidden images never load lazily
      if (img.complete) done(img, img.naturalWidth > 0);
      else {
        img.addEventListener('load', () => done(img, true), { once: true });
        img.addEventListener('error', () => done(img, false), { once: true });
      }
    });
  }

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
