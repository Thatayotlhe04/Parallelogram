/* parallelogram.dev — landing interactions
   Three things happen here:
   1. The terminal demo "video" — typed commands and streaming output on loop.
   2. The hero install chip — click to copy.
   3. Scroll reveals via IntersectionObserver.
*/

// ── 1. terminal demo ────────────────────────────────────────────────
const term = document.getElementById('terminal-body');

const seq = [
  { t: 'cmd',     text: 'parallelogram check data.jsonl --tokenizer meta-llama/Llama-3-8B' },
  { t: 'wait',    ms: 450 },
  { t: 'err',     line: 23,  rule: 'roles',          msg: "Conversation must end on 'assistant', ended on 'user'" },
  { t: 'wait',    ms: 90 },
  { t: 'err',     line: 147, rule: 'duplicates',     msg: 'Duplicate of line 89' },
  { t: 'wait',    ms: 90 },
  { t: 'err',     line: 312, rule: 'context-window', msg: 'Record exceeds max_seq_len: ~8512 > 8192 tokens' },
  { t: 'wait',    ms: 90 },
  { t: 'warn',    line: 401, rule: 'encoding',       msg: "Message 0 contains likely mojibake: 'â\u20AC™'" },
  { t: 'blank' },
  { t: 'summary', text: '547 records  543 clean  3 errors  1 warnings' },
  { t: 'wait',    ms: 1900 },
  { t: 'clear' },
  { t: 'cmd',     text: 'parallelogram check data.jsonl --output clean.jsonl' },
  { t: 'wait',    ms: 450 },
  { t: 'ok',      text: '✓ Clean: 547 records validated' },
  { t: 'note',    text: '→ Wrote 547 clean records to clean.jsonl' },
  { t: 'wait',    ms: 2400 },
  { t: 'clear' },
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

function appendLine(cls, html) {
  const el = document.createElement('span');
  el.className = `t-line ${cls}`;
  el.innerHTML = html;
  term.appendChild(el);
  return el;
}

async function typeCmd(text) {
  const el = appendLine('cmd', '<span class="pr">$</span>');
  const cur = document.createElement('span');
  cur.className = 'cur';
  el.appendChild(cur);
  for (const ch of text) {
    cur.insertAdjacentText('beforebegin', ch);
    // small jitter so it feels human-typed
    await sleep(18 + Math.random() * 26);
  }
  await sleep(180);
  cur.remove();
}

function clearTerm() { term.innerHTML = ''; }

function fmtIssue(line, rule, msg, mark, markCls) {
  return `  <span class="${markCls}">${mark}</span> data.jsonl:<span class="lno">${line}</span> <span class="rid">[${rule}]</span> ${msg}`;
}

async function runTerminal() {
  // run once to start (no clear flicker on first paint)
  while (true) {
    for (const step of seq) {
      switch (step.t) {
        case 'cmd':     await typeCmd(step.text); break;
        case 'err':     appendLine('err',     fmtIssue(step.line, step.rule, step.msg, '✗', 'err')); break;
        case 'warn':    appendLine('warn',    fmtIssue(step.line, step.rule, step.msg, '!', 'warn')); break;
        case 'ok':      appendLine('ok',      `  ${step.text}`); break;
        case 'note':    appendLine('note',    `  ${step.text}`); break;
        case 'summary': appendLine('summary', step.text); break;
        case 'blank':   appendLine('blank', '&nbsp;'); break;
        case 'wait':    await sleep(step.ms); break;
        case 'clear':   clearTerm(); break;
      }
    }
  }
}

// only animate when the terminal is on screen — saves CPU when scrolled past
let started = false;
const startWhenVisible = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting && !started) {
      started = true;
      runTerminal();
      startWhenVisible.disconnect();
    }
  }
}, { threshold: 0.2 });
startWhenVisible.observe(term);

// ── 2. install-chip copy-to-clipboard ───────────────────────────────
const chip = document.getElementById('install-chip');
chip?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('parallelogram check data.jsonl');
    chip.classList.add('copied');
    setTimeout(() => chip.classList.remove('copied'), 1600);
  } catch {
    // clipboard API can be blocked; silently fail rather than break the page
  }
});

// ── 3. reveal-on-scroll ─────────────────────────────────────────────
const reveals = document.querySelectorAll('[data-reveal]');
const revealObs = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add('in');
      revealObs.unobserve(e.target);
    }
  }
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
reveals.forEach(el => revealObs.observe(el));

// ── 4. compare demo (input vs output stream) ────────────────────────
/*
  Streams rows into both panes on a loop. The "in" pane shows raw records
  with three errors and a warning interleaved among 18 valid rows; the
  "out" pane shows only the valid rows in the same order, so the eye can
  see the exact records being held back.
*/
const inPane  = document.getElementById('compare-stream-in');
const outPane = document.getElementById('compare-stream-out');

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

  async function runCompare() {
    // start fresh every loop
    while (true) {
      inPane.innerHTML = '';
      outPane.innerHTML = '';
      const inCounter  = document.getElementById('cm-in');
      const outCounter = document.getElementById('cm-out');
      if (inCounter)  inCounter.textContent  = '0';
      if (outCounter) outCounter.textContent = '0';

      let totalIn = 0;
      let totalOut = 0;

      for (const [line, kind, label, tag] of dataset) {
        const row = makeRow(line, kind, label, tag);
        inPane.appendChild(row);
        // keep only the latest 8 rows visible
        while (inPane.children.length > 8) inPane.firstChild.remove();
        totalIn++;
        if (inCounter) inCounter.textContent = totalIn * 39; // scale up for drama (547 by end)

        // valid rows also appear on the right, slightly delayed
        if (kind === 'good' || kind === 'warn') {
          await sleep(220);
          const outRow = makeRow(line, kind, label, tag);
          outPane.appendChild(outRow);
          while (outPane.children.length > 8) outPane.firstChild.remove();
          totalOut++;
          if (outCounter) outCounter.textContent = totalOut * 39;
        } else {
          // errors: pause briefly, no echo to the clean side
          await sleep(380);
        }
      }

      // settle on final tallies
      if (inCounter)  inCounter.textContent  = '547';
      if (outCounter) outCounter.textContent = '543';
      await sleep(2400);
    }
  }

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

// ── 5. exit-code cycler ─────────────────────────────────────────────
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

// ── 6. stat counters ────────────────────────────────────────────────
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
    if (target === 0) {
      el.textContent = '0';
      counterObs.unobserve(el);
      continue;
    }
    const start = performance.now();
    const duration = 1200;
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(target * eased).toString();
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    counterObs.unobserve(el);
  }
}, { threshold: 0.4 });
counters.forEach(c => counterObs.observe(c));
