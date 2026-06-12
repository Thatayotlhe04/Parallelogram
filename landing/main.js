/* parallelogram.dev — landing interactions
   The hero terminal lives in hero-terminal.js and the live demo in demo.js.
   This file owns everything else: header state, mobile nav, reveals,
   scroll-spy, copy buttons, the compare stream, the exit-code cycler,
   stat counters, the token meter, the architecture inspector, and the
   cookie banner. No dependencies, no GSAP — IntersectionObserver and rAF
   cover everything. Every hook is null-checked because privacy.html and
   terms.html load this file with most of these elements absent.
*/
(() => {
  'use strict';

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const prefersReducedMotion =
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const header = document.getElementById('site-header');

  // ── 2. header scroll state ──────────────────────────────────────────
  /*
    A 1px sentinel at the top of <body> tells us whether the page has
    scrolled without a scroll listener: sentinel out of view ⇒ scrolled.
  */
  (() => {
    const sentinel = document.getElementById('scroll-sentinel');
    if (!sentinel || !header) return;
    new IntersectionObserver((entries) => {
      for (const e of entries) {
        header.classList.toggle('is-scrolled', !e.isIntersecting);
      }
    }).observe(sentinel);
  })();

  // ── 3. mobile nav ───────────────────────────────────────────────────
  (() => {
    const toggle = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    if (!toggle || !header) return;

    function setOpen(open) {
      header.classList.toggle('is-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    toggle.addEventListener('click', () =>
      setOpen(!header.classList.contains('is-open')));

    // picking a destination closes the panel; Escape closes and hands
    // focus back to the toggle so keyboard users aren't stranded
    menu?.addEventListener('click', (e) => {
      if (e.target.closest('a')) setOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && header.classList.contains('is-open')) {
        setOpen(false);
        toggle.focus();
      }
    });
  })();

  // ── 4. reveal-on-scroll ─────────────────────────────────────────────
  /*
    An IntersectionObserver toggles .in; the CSS transition (and any
    --i stagger delay) does the visual work. Reduced-motion users just
    see content already there.
  */
  (() => {
    const reveals = Array.from(document.querySelectorAll('[data-reveal]'));
    if (!reveals.length) return;
    if (prefersReducedMotion) {
      reveals.forEach(el => el.classList.add('in'));
      return;
    }
    const pending = new Set(reveals);
    const revealObs = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) mark(e.target);
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    function mark(el) {
      el.classList.add('in');
      pending.delete(el);
      revealObs.unobserve(el);
    }
    reveals.forEach(el => revealObs.observe(el));

    /* Backstop: the observer can miss elements inside
       content-visibility:auto sections when the page jumps instantly
       (deep links, anchor clicks) — Chromium reports stale rects until
       the next layout. A rAF-throttled sweep marks anything the
       viewport already contains, then unhooks itself when done. */
    let raf = 0;
    function sweep() {
      raf = 0;
      for (const el of pending) {
        const r = el.getBoundingClientRect();
        if ((r.width || r.height) && r.top < innerHeight - 40 && r.bottom > 0) {
          mark(el);
        }
      }
      if (!pending.size) {
        removeEventListener('scroll', onScroll);
        removeEventListener('hashchange', onScroll);
      }
    }
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(sweep); };
    addEventListener('scroll', onScroll, { passive: true });
    addEventListener('hashchange', onScroll);
    sweep();
  })();

  // ── 5. nav scroll-spy ───────────────────────────────────────────────
  /*
    Marks the nav link for whichever section currently owns the viewport,
    so the underline reflects where you are without any scroll math.
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

  // ── 6. copy buttons ─────────────────────────────────────────────────
  /*
    Any [data-copy] copies its payload. The label swap is per-button; the
    aria-live announcement is shared so screen readers hear one message
    no matter which button fired. Icon-only buttons have no .copy-label.
  */
  (() => {
    const announce = document.getElementById('copy-announce');
    let announceTimer = 0;

    document.querySelectorAll('[data-copy]').forEach((btn) => {
      const label = btn.querySelector('.copy-label');
      const original = label ? label.textContent : '';
      let restoreTimer = 0;

      btn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(btn.dataset.copy || '');
          btn.classList.add('copied');
          if (label) label.textContent = 'Copied';
          clearTimeout(restoreTimer);
          restoreTimer = setTimeout(() => {
            btn.classList.remove('copied');
            if (label) label.textContent = original;
          }, 1600);
          if (announce) {
            announce.textContent = 'Copied to clipboard';
            clearTimeout(announceTimer);
            announceTimer = setTimeout(() => { announce.textContent = ''; }, 2000);
          }
        } catch {
          // clipboard API can be blocked; silently fail rather than break the page
        }
      });
    });
  })();

  // ── 7. compare demo (input vs output stream) ────────────────────────
  /*
    Streams rows into both panes on a loop. The "in" pane shows raw records
    with three errors and a warning interleaved among valid rows; the "out"
    pane shows only the rows that survive, so the eye can see exactly which
    records are held back (each flashes red before it's dropped). A replay
    control restarts the cycle from the top. Reduced motion paints the
    final frame once instead of looping.
  */
  (() => {
    const inPane  = document.getElementById('compare-stream-in');
    const outPane = document.getElementById('compare-stream-out');
    const compareReplay = document.getElementById('compare-replay');
    if (!inPane || !outPane) return;

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

    if (prefersReducedMotion) {
      // one static frame: every row in, survivors out, final counters
      function renderFinal() {
        inPane.innerHTML = '';
        outPane.innerHTML = '';
        for (const [line, kind, label, tag] of dataset) {
          inPane.appendChild(makeRow(line, kind, label, tag));
          if (kind === 'good' || kind === 'warn') {
            outPane.appendChild(makeRow(line, kind, label, tag));
          }
        }
        const inCounter  = document.getElementById('cm-in');
        const outCounter = document.getElementById('cm-out');
        if (inCounter)  inCounter.textContent  = '547';
        if (outCounter) outCounter.textContent = '543';
        compareReplay?.removeAttribute('hidden');
      }
      compareReplay?.addEventListener('click', renderFinal);
      let done = false;
      const obs = new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (e.isIntersecting && !done) {
            done = true;
            renderFinal();
            obs.disconnect();
          }
        }
      }, { threshold: 0.2 });
      obs.observe(inPane);
      return;
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
  })();

  // ── 8. exit-code cycler ─────────────────────────────────────────────
  /*
    Highlights one of the three exit codes at a time, on a slow rotation,
    so the visual eye can land on the meaning rather than reading three
    pills cold. The interval only runs while the pills are on screen;
    reduced motion pins the highlight on exit 0.
  */
  (() => {
    const ecPills = ['ec-0', 'ec-1', 'ec-2']
      .map(id => document.getElementById(id)).filter(Boolean);
    if (ecPills.length !== 3) return;

    if (prefersReducedMotion) {
      ecPills[0].classList.add('is-active');
      return;
    }

    let idx = 0;
    let timer = 0;
    function cycleExitCode() {
      ecPills.forEach(p => p.classList.remove('is-active'));
      ecPills[idx].classList.add('is-active');
      idx = (idx + 1) % ecPills.length;
    }
    cycleExitCode();

    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !timer) {
          timer = setInterval(cycleExitCode, 1800);
        } else if (!e.isIntersecting && timer) {
          clearInterval(timer);
          timer = 0;
        }
      }
    });
    io.observe(ecPills[0]);
  })();

  // ── 9. stat counters ────────────────────────────────────────────────
  /*
    Each [data-counter] element knows its target value. When the stats
    band scrolls into view, count up from 0 to the target over ~1.2s.
  */
  (() => {
    const counters = document.querySelectorAll('[data-counter]');
    if (!counters.length) return;
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
  })();

  // ── 10. cookie consent banner ───────────────────────────────────────
  (() => {
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
  })();

  // ── 11. token meter ─────────────────────────────────────────────────
  /*
    Counts one record's tokens up to 5,234 against a 4,096 budget. The
    track represents 0–5500 so the max_seq_len marker sits inside the bar
    (4096/5500 ≈ 74.5%) and the fill visibly overshoots it. Crossing the
    budget flips the status pill to ERROR and reveals the diagnostic —
    the same moment the CLI would flag context-window.
  */
  (() => {
    const meter = document.getElementById('token-meter');
    const fill = document.getElementById('meter-fill');
    const count = document.getElementById('meter-count');
    const status = document.getElementById('meter-status');
    const diag = document.getElementById('meter-diag');
    const replay = document.getElementById('meter-replay');
    if (!meter || !fill || !count) return;

    const TARGET = 5234;     // this record's token count
    const LIMIT = 4096;      // --max-seq-len default
    const SCALE = 5500;      // track top end
    const DURATION = 2200;
    let meterToken = 0;      // bumps to cancel an in-flight rAF loop on replay

    function setCount(n) {
      count.textContent = n.toLocaleString('en-US');
      fill.style.setProperty('--fill',
        String(Math.min(1, Math.max(0, n / SCALE))));
    }

    function setOver(over) {
      meter.classList.toggle('is-over', over);
      if (status) {
        status.textContent = over ? 'ERROR · over budget' : 'EXACT · o200k_base';
        status.className = over ? 'pill pill--error' : 'pill pill--info';
      }
      if (diag) diag.toggleAttribute('hidden', !over);
    }

    function run() {
      const myToken = ++meterToken;
      if (prefersReducedMotion) {
        setCount(TARGET);
        setOver(true);
        return;
      }
      setOver(false);
      setCount(0);
      const start = performance.now();
      function tick(now) {
        if (myToken !== meterToken) return;
        const t = Math.min(1, (now - start) / DURATION);
        const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
        const n = Math.round(TARGET * eased);
        setCount(n);
        if (n > LIMIT && !meter.classList.contains('is-over')) setOver(true);
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    replay?.addEventListener('click', run);

    let started = false;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started) {
          started = true;
          run();
          io.disconnect();
        }
      }
    }, { threshold: 0.4 });
    io.observe(meter);
  })();

  // ── 12. architecture inspector ──────────────────────────────────────
  /*
    Each visual node explains itself in #arch-desc on click; clicking
    the active node again restores the default prompt. The .in class on
    .arch arms the route animation once the diagram is on screen.
  */
  (() => {
    const arch = document.querySelector('.arch');
    const desc = document.getElementById('arch-desc');
    const nodes = Array.from(document.querySelectorAll('button.arch-hotspot[data-arch]'));

    const COPY = {
      openai: 'OpenAI chat JSONL and ShareGPT enter as raw files; the format branch shows they can change without forcing rule-specific forks.',
      parser: 'The parse boundary turns every supported format into one internal message shape before any rule runs.',
      structural: 'Schema, role order, empty content, and duplicate checks run on an isolated preview path before records can return to the baseline.',
      context: 'Tokenizer-aware limits are tested as their own branch: exact when a tokenizer is available, estimated when it is not.',
      fixer: 'The safe fixer can repair mechanical issues, then re-validates the result before anything is allowed through.',
      output: 'Only records that pass every rule return to the clean output path and can be written to clean.jsonl.',
      dropped: 'Records that still fail after safe repair are dropped with a reason instead of silently contaminating the training set.',
    };

    const defaultDesc = desc ? desc.textContent : '';

    nodes.forEach((node) => {
      node.addEventListener('click', () => {
        const wasActive = node.classList.contains('is-active');
        nodes.forEach((n) => {
          n.classList.remove('is-active');
          n.setAttribute('aria-pressed', 'false');
        });
        if (wasActive) {
          if (desc) desc.textContent = defaultDesc;
          return;
        }
        node.classList.add('is-active');
        node.setAttribute('aria-pressed', 'true');
        if (desc) desc.textContent = COPY[node.dataset.arch] || defaultDesc;
      });
    });

    if (!arch) return;

    if (prefersReducedMotion) {
      arch.classList.add('in');
      return;
    }
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          arch.classList.add('in');
          io.disconnect();
        }
      }
    }, { threshold: 0.2 });
    io.observe(arch);
  })();
})();
