/* parallelogram.dev — landing interactions
   The interactive terminal demo lives in demo.js. This file handles the
   surrounding motion and micro-interactions:
   1. Motion foundation — GSAP-driven reveals (with an IntersectionObserver
      fallback), hero parallax, magnetic buttons, brand draw-on, scroll-spy.
   2. The hero install chip — click to copy.
   3. The compare demo (input vs output stream) + replay control.
   4. Exit-code cycler, stat counters, cookie banner.
   Everything degrades gracefully without GSAP and respects reduced-motion.
*/

const sleep = ms => new Promise(r => setTimeout(r, ms));
const prefersReducedMotion =
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const hasGSAP = !!window.gsap && !prefersReducedMotion;
if (hasGSAP && window.ScrollTrigger) gsap.registerPlugin(ScrollTrigger);

// ── 1. reveal-on-scroll ─────────────────────────────────────────────
/*
  With GSAP present we batch the reveals so a cluster entering the viewport
  staggers in; otherwise an IntersectionObserver toggles the same .in class.
  Both paths rely on the CSS transition on .reveal, so the visual is identical
  bar the stagger — and reduced-motion users just see content appear.
*/
const reveals = Array.from(document.querySelectorAll('[data-reveal]'));
if (prefersReducedMotion) {
  reveals.forEach(el => el.classList.add('in'));
} else if (hasGSAP && window.ScrollTrigger) {
  ScrollTrigger.batch('[data-reveal]', {
    start: 'top 88%',
    onEnter: batch => batch.forEach((el, i) =>
      setTimeout(() => el.classList.add('in'), i * 90)),
  });
} else {
  const revealObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        revealObs.unobserve(e.target);
      }
    }
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  reveals.forEach(el => revealObs.observe(el));
}

// ── 2. hero parallax ────────────────────────────────────────────────
/*
  Subtle pointer-reactive drift on the hero grid and headline. Throttled to
  one update per frame; only runs with GSAP, a fine pointer, and motion on.
*/
(() => {
  if (!hasGSAP) return;
  if (!window.matchMedia('(pointer: fine)').matches) return;
  const hero = document.querySelector('.hero');
  const grid = document.querySelector('.grid-bg');
  const headline = document.querySelector('.headline');
  if (!hero || !grid) return;

  let raf = 0, mx = 0, my = 0;
  hero.addEventListener('pointermove', (e) => {
    const r = hero.getBoundingClientRect();
    mx = (e.clientX - r.left) / r.width - 0.5;   // -0.5 … 0.5
    my = (e.clientY - r.top) / r.height - 0.5;
    if (!raf) raf = requestAnimationFrame(apply);
  });
  hero.addEventListener('pointerleave', () => { mx = 0; my = 0; if (!raf) raf = requestAnimationFrame(apply); });

  function apply() {
    raf = 0;
    gsap.to(grid, { x: mx * 18, y: my * 18, duration: 0.6, ease: 'power2.out' });
    if (headline) gsap.to(headline, { x: mx * 6, y: my * 5, duration: 0.6, ease: 'power2.out' });
  }
})();

// ── 3. magnetic buttons ─────────────────────────────────────────────
/*
  Primary buttons lean toward the cursor when hovered, capped so it stays
  tasteful. Pure transform, reset on leave. Skipped without GSAP / on touch.
*/
(() => {
  if (!hasGSAP) return;
  if (!window.matchMedia('(pointer: fine)').matches) return;
  document.querySelectorAll('.btn-primary').forEach((btn) => {
    btn.addEventListener('pointermove', (e) => {
      const r = btn.getBoundingClientRect();
      const x = (e.clientX - r.left - r.width / 2) * 0.3;
      const y = (e.clientY - r.top - r.height / 2) * 0.4;
      gsap.to(btn, { x: Math.max(-7, Math.min(7, x)), y: Math.max(-5, Math.min(5, y)), duration: 0.3, ease: 'power2.out' });
    });
    btn.addEventListener('pointerleave', () => {
      gsap.to(btn, { x: 0, y: 0, duration: 0.4, ease: 'elastic.out(1, 0.4)' });
    });
  });
})();

// ── 4. brand mark draw-on ───────────────────────────────────────────
/*
  The parallelogram outline traces itself once on load. Reduced-motion users
  see the finished mark immediately (we never touch it).
*/
(() => {
  if (prefersReducedMotion) return;
  const paths = document.querySelectorAll('.cta-glyph path, .hero .brand-mark path');
  paths.forEach((path) => {
    const len = path.getTotalLength ? path.getTotalLength() : 0;
    if (!len) return;
    path.style.strokeDasharray = len;
    path.style.strokeDashoffset = len;
    if (hasGSAP) {
      gsap.to(path, { strokeDashoffset: 0, duration: 1.4, ease: 'power2.inOut',
        scrollTrigger: window.ScrollTrigger ? { trigger: path, start: 'top 92%' } : undefined });
    } else {
      // CSS-less fallback: just clear it after a tick so it's never stuck hidden
      requestAnimationFrame(() => { path.style.transition = 'stroke-dashoffset 1.4s ease'; path.style.strokeDashoffset = 0; });
    }
  });
})();

// ── 5. nav scroll-spy ───────────────────────────────────────────────
/*
  Marks the nav link for whichever section currently owns the viewport, so
  the underline reflects where you are without any scroll math.
*/
(() => {
  const links = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  if (!links.length) return;
  const map = new Map();
  links.forEach((a) => {
    const id = a.getAttribute('href').slice(1);
    const sec = document.getElementById(id);
    if (sec) map.set(sec, a);
  });
  if (!map.size) return;

  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        links.forEach(a => a.removeAttribute('aria-current'));
        map.get(e.target)?.setAttribute('aria-current', 'true');
      }
    }
  }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
  map.forEach((_, sec) => spy.observe(sec));
})();

// ── 6. install-chip copy-to-clipboard ───────────────────────────────
const chip = document.getElementById('install-chip');
chip?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('pip install parallelogram');
    chip.classList.add('copied');
    setTimeout(() => chip.classList.remove('copied'), 1600);
  } catch {
    // clipboard API can be blocked; silently fail rather than break the page
  }
});

// ── 7. compare demo (input vs output stream) ────────────────────────
/*
  Streams rows into both panes on a loop. The "in" pane shows raw records
  with three errors and a warning interleaved among valid rows; the "out"
  pane shows only the rows that survive, so the eye can see exactly which
  records are held back (each flashes red before it's dropped). A replay
  control restarts the cycle from the top.
*/
const inPane  = document.getElementById('compare-stream-in');
const outPane = document.getElementById('compare-stream-out');
const compareReplay = document.getElementById('compare-replay');

if (inPane && outPane) {
  // [line, kind, label, badge?]
  const dataset = [
    [1,   'good', '{"messages":[…]}'],
    [2,   'good', '{"messages":[…]}'],
    [23,  'bad',  '{"messages":[…]}', 'roles'],
    [24,  'good', '{"messages":[…]}'],
    [25,  'good', '{"messages":[…]}'],
    [89,  'good', '{"messages":[…]}'],
    [147, 'bad',  '{"messages":[…]}', 'duplicates'],
    [148, 'good', '{"messages":[…]}'],
    [200, 'good', '{"messages":[…]}'],
    [312, 'bad',  '{"messages":[…]}', 'context-window'],
    [313, 'good', '{"messages":[…]}'],
    [401, 'warn', '{"messages":[…]}', 'encoding'],
    [402, 'good', '{"messages":[…]}'],
    [500, 'good', '{"messages":[…]}'],
  ];

  function makeRow(line, kind, label, tag) {
    const div = document.createElement('div');
    div.className = `cs-row ${kind}`;
    div.innerHTML = `<span class="cs-line">L${line}</span><span class="cs-body">${label}${tag ? `<span class="cs-tag">${tag}</span>` : ''}</span>`;
    return div;
  }

  let compareToken = 0;

  async function runCompare() {
    const myToken = ++compareToken;
    while (myToken === compareToken) {
      inPane.innerHTML = '';
      outPane.innerHTML = '';
      const inCounter  = document.getElementById('cm-in');
      const outCounter = document.getElementById('cm-out');
      if (inCounter)  inCounter.textContent  = '0';
      if (outCounter) outCounter.textContent = '0';

      let totalIn = 0;
      let totalOut = 0;

      for (const [line, kind, label, tag] of dataset) {
        if (myToken !== compareToken) return;
        const row = makeRow(line, kind, label, tag);
        inPane.appendChild(row);
        while (inPane.children.length > 8) inPane.firstChild.remove();
        totalIn++;
        if (inCounter) inCounter.textContent = totalIn * 39; // scale up for drama (547 by end)

        if (kind === 'good' || kind === 'warn') {
          // valid rows (and recoverable warnings) echo to the clean side
          await sleep(220);
          const outRow = makeRow(line, kind, label, tag);
          outPane.appendChild(outRow);
          while (outPane.children.length > 8) outPane.firstChild.remove();
          totalOut++;
          if (outCounter) outCounter.textContent = totalOut * 39;
        } else {
          // errors: flash the held-back row, then pause — no echo to clean side
          row.classList.add('flash-out');
          await sleep(380);
        }
      }

      if (inCounter)  inCounter.textContent  = '547';
      if (outCounter) outCounter.textContent = '543';
      compareReplay?.removeAttribute('hidden');
      await sleep(2600);
    }
  }

  compareReplay?.addEventListener('click', () => runCompare());

  let compareStarted = false;
  const compareObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !compareStarted) {
        compareStarted = true;
        runCompare();
        compareObs.disconnect();
      }
    }
  }, { threshold: 0.2 });
  compareObs.observe(inPane);
}

// ── 8. exit-code cycler ─────────────────────────────────────────────
/*
  Highlights one of the three exit codes at a time, on a slow rotation,
  so the visual eye can land on the meaning rather than reading three
  pills cold.
*/
const ecPills = ['ec-0', 'ec-1', 'ec-2'].map(id => document.getElementById(id)).filter(Boolean);
if (ecPills.length === 3) {
  let idx = 0;
  function cycleExitCode() {
    ecPills.forEach(p => p.classList.remove('is-active'));
    ecPills[idx].classList.add('is-active');
    idx = (idx + 1) % ecPills.length;
  }
  cycleExitCode();
  setInterval(cycleExitCode, 1800);
}

// ── 9. stat counters ────────────────────────────────────────────────
/*
  Each [data-counter] element knows its target value. When the stats
  band scrolls into view, count up from 0 to the target over ~1.2s.
*/
const counters = document.querySelectorAll('[data-counter]');
const counterObs = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const el = e.target;
    const target = parseInt(el.dataset.counter, 10);
    if (target === 0 || prefersReducedMotion) {
      el.textContent = String(target);
      counterObs.unobserve(el);
      continue;
    }
    const start = performance.now();
    const duration = 1200;
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      el.textContent = Math.round(target * eased).toString();
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    counterObs.unobserve(el);
  }
}, { threshold: 0.4 });
counters.forEach(c => counterObs.observe(c));

// ── 10. cookie consent banner ───────────────────────────────────────
const cookieBanner = document.getElementById('cookie-banner');
const cookieAccept = document.getElementById('cookie-accept');
const cookieDecline = document.getElementById('cookie-decline');

const CONSENT_KEY = 'parallelogram_cookie_consent';
const CONSENT_COOKIE = 'pg_cookie_consent';

function setConsentCookie(value) {
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${CONSENT_COOKIE}=${value}; max-age=${maxAge}; path=/; SameSite=Lax`;
}

function storeConsent(value) {
  localStorage.setItem(CONSENT_KEY, value);
  setConsentCookie(value);
  cookieBanner?.setAttribute('hidden', '');
}

if (cookieBanner && !localStorage.getItem(CONSENT_KEY)) {
  cookieBanner.removeAttribute('hidden');
}

cookieAccept?.addEventListener('click', () => storeConsent('accepted'));
cookieDecline?.addEventListener('click', () => storeConsent('declined'));
