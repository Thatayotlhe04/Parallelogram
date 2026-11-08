/* parallelogram.dev — copy buttons, CLI/CI tabs, Human/Agent views, and the eater.
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

  // ── Nav menus ─────────────────────────────────────────────────────
  // CSS opens a menu on hover. This adds click and keyboard (a disclosure
  // button), keeps aria-expanded truthful, and closes on Escape or outside.
  const items = Array.from(document.querySelectorAll('.nav-item.has-menu'));
  const hoverable = window.matchMedia('(hover: hover)');
  function setOpen(item, open) {
    item.classList.toggle('open', open);
    item.querySelector('.nav-trigger').setAttribute('aria-expanded', String(open));
  }
  function closeAll(except) { items.forEach(i => { if (i !== except) setOpen(i, false); }); }
  items.forEach(item => {
    const trigger = item.querySelector('.nav-trigger');
    trigger.addEventListener('click', () => {
      const open = !item.classList.contains('open');
      closeAll(item);
      setOpen(item, open);
    });
    item.addEventListener('mouseenter', () => {
      if (!hoverable.matches) return;
      closeAll(item);
      trigger.setAttribute('aria-expanded', 'true');
    });
    item.addEventListener('mouseleave', () => {
      if (hoverable.matches) setOpen(item, false);
    });
    item.addEventListener('focusout', e => {
      if (!item.contains(e.relatedTarget)) setOpen(item, false);
    });
    item.addEventListener('keydown', e => {
      if (e.key !== 'Escape') return;
      setOpen(item, false);
      trigger.focus();
    });
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.nav-item.has-menu')) closeAll();
  });

  // Mobile menu: close it after following an in-page link, and on Escape.
  const mnav = document.querySelector('.mnav');
  if (mnav) {
    mnav.addEventListener('click', e => { if (e.target.closest('a')) mnav.open = false; });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && mnav.open) { mnav.open = false; mnav.querySelector('summary').focus(); }
    });
    window.matchMedia('(min-width: 961px)').addEventListener('change', e => { if (e.matches) mnav.open = false; });
  }

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

  // ── Human / Agent ─────────────────────────────────────────────────
  // Agent shows the machine-readable page (landing/index.md, embedded by
  // scripts/build_agent_view.py) in place of the designed one. The choice
  // is remembered per browser; ?view=agent links straight to it.
  const modeButtons = Array.from(document.querySelectorAll('.mode [data-mode]'));
  const agentView = document.getElementById('agent-view');
  const agentMd = document.getElementById('agent-md');
  const markdown = agentMd ? agentMd.textContent : '';
  let highlighted = false;

  const esc = text => text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const inline = text => esc(text)
    .replace(/`([^`]+)`/g, '<span class="md-code">`$1`</span>')
    .replace(/(https?:\/\/[^\s<)]+)/g, '<span class="md-link">$1</span>');
  function highlight() {
    if (highlighted || !agentMd) return;
    let fence = false, front = 0;
    agentMd.innerHTML = markdown.split('\n').map(line => {
      if (line === '---' && front < 2) { front++; return '<span class="md-dim">---</span>'; }
      if (front === 1) return '<span class="md-dim">' + esc(line) + '</span>';
      if (line.startsWith('```')) { fence = !fence; return '<span class="md-dim">' + esc(line) + '</span>'; }
      if (fence) return '<span class="md-pre">' + esc(line) + '</span>';
      const h = line.match(/^(#{1,6}) (.*)$/);
      if (h) return '<span class="md-mark">' + h[1] + '</span> <span class="md-h">' + inline(h[2]) + '</span>';
      if (line.startsWith('> ')) return '<span class="md-dim">' + inline(line) + '</span>';
      const li = line.match(/^(\s*- )(.*)$/);
      if (li) return '<span class="md-mark">' + li[1] + '</span>' + inline(li[2]);
      return inline(line);
    }).join('\n');
    highlighted = true;
  }

  function setMode(mode, remember) {
    if (!agentView) return;
    const agent = mode === 'agent';
    if (agent) highlight();
    document.documentElement.dataset.mode = agent ? 'agent' : 'human';
    agentView.hidden = !agent;
    modeButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    if (remember) {
      try { localStorage.setItem('pg-mode', mode); } catch { /* storage unavailable */ }
      window.scrollTo(0, 0);
    }
  }
  modeButtons.forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode, true)));
  let initial = 'human';
  try { initial = localStorage.getItem('pg-mode') || 'human'; } catch { /* storage unavailable */ }
  if (new URLSearchParams(location.search).get('view') === 'agent') initial = 'agent';
  if (initial === 'agent') setMode('agent', false);

  const copyPage = document.getElementById('agent-copy');
  if (copyPage) copyPage.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(markdown);
      copyPage.textContent = 'Copied';
    } catch {
      copyPage.textContent = 'Select and copy';
    }
    setTimeout(() => { copyPage.textContent = 'Copy page'; }, 2000);
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
