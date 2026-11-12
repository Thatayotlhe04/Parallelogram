/* A paced walkthrough of a real CLI run and its actual input/output files. */
(() => {
  'use strict';
  const SCENES = [
    ['intro', 'YOUR TRAINING DATA', 'Small file. Hidden problems.', 'A repeated example. A private email. Both easy to miss in a training set.', '3 training examples · customer support'],
    ['duplicate', 'DUPLICATE FOUND', 'You’ve seen this one before.', 'Record 2 repeats record 1. Parallelogram points to the exact duplicate.', 'train.jsonl:2 — duplicate of record 1'],
    ['private', 'PRIVATE DATA FOUND', 'That email doesn’t belong here.', 'A personal email is hiding in an answer. Catch it before it becomes training data.', 'train.jsonl:3 — email address detected'],
    ['fix', 'WRITE A CLEAN COPY', 'One command. Both handled.', 'Remove the duplicate and redact the email. Keep the original file untouched.', 'Writing fixes to clean.jsonl…'],
    ['clean', 'THE RESULT', 'Less repetition. No private email.', 'Two examples remain. The email is replaced with a placeholder in the new file.', 'clean.jsonl — 1 duplicate removed · 1 email redacted'],
    ['done', 'VALIDATION PASSED', 'Checked. Fixed. Checked again.', 'The new file passes with zero errors and zero warnings. Now continue your training workflow.', 'clean.jsonl — 2 records · 0 errors · 0 warnings'],
  ];
  function timeline(commands) {
    const starts = [1800, 11800, 19000];
    return commands.map((command, i) => ({start: starts[i], typedAt: starts[i] + command.length * 32,
      outputAt: starts[i] + command.length * 32 + 500}));
  }
  const DURATION = 28000;
  function frameAt(time, steps) {
    const elapsed = Math.max(0, Number(time) || 0);
    const stage = elapsed >= steps[2].start ? 2 : elapsed >= steps[1].start ? 1 : 0;
    const step = steps[stage];
    const scene = elapsed >= steps[2].outputAt ? 5 : elapsed >= 15800 ? 4 :
      elapsed >= steps[1].start ? 3 : elapsed >= 7700 ? 2 : elapsed >= steps[0].outputAt ? 1 : 0;
    return {stage, scene, typed: Math.max(0, Math.min(1, (elapsed-step.start)/(step.typedAt-step.start))),
      output: elapsed >= step.outputAt, reset: elapsed >= DURATION - 650};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {timeline, frameAt, DURATION, SCENES};
  if (typeof document === 'undefined') return;
  const byId = id => document.getElementById(id);
  const desktop = byId('demo-desktop');
  if (!desktop) return;
  const screen = byId('terminal-screen'), pane = byId('pane-cli'), transcript = byId('terminal-static');
  const runs = Array.from(screen.querySelectorAll('[data-run]')).map(element => ({
    element, command: element.querySelector('[data-command]'), output: element.querySelector('[data-output]'),
  }));
  const steps = timeline(runs.map(run => run.command.dataset.command));
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let time = 0, playing = false, inView = false, reading = false, previous = null, raf = 0;
  let lastScene = -1, lastStage = -1, lastOutput = false;
  function render() {
    const state = frameAt(time, steps);
    desktop.classList.toggle('is-running', playing);
    desktop.classList.toggle('is-resetting', state.reset);
    runs.forEach((run, index) => {
      run.element.hidden = index > state.stage;
      const full = run.command.dataset.command;
      const text = index < state.stage ? full : index === state.stage ? full.slice(0, Math.ceil(full.length * state.typed)) : '';
      if (run.command.textContent !== text) run.command.textContent = text;
      run.output.hidden = !(index < state.stage || index === state.stage && state.output);
      run.element.classList.toggle('is-typing', index === state.stage && !state.output);
    });
    byId('terminal-idle').hidden = !state.output;
    if (lastScene !== state.scene) {
      const [id, label, title, body, note] = SCENES[state.scene];
      desktop.dataset.scene = id;
      byId('story-label').textContent = label;
      byId('story-title').textContent = title;
      byId('story-body').textContent = body;
      byId('editor-note').textContent = note;
      const clean = state.scene >= 4;
      byId('source-before').hidden = clean;
      byId('source-after').hidden = !clean;
      byId('editor-filename').textContent = clean ? 'clean.jsonl' : 'train.jsonl';
      byId('editor-status').textContent = clean ? '2 records  ·  JSONL' : '3 records  ·  JSONL';
    }
    // Confine terminal scrolling to new output; never move the surrounding page.
    if (playing && (state.stage !== lastStage || state.output !== lastOutput)) screen.scrollTop = screen.scrollHeight;
    lastScene = state.scene; lastStage = state.stage; lastOutput = state.output;
  }
  function pause() { playing = false; previous = null; cancelAnimationFrame(raf); render(); }
  function tick(now) {
    if (!playing) return;
    if (previous !== null) time += now - previous;
    previous = now;
    if (time >= DURATION) { time %= DURATION; lastScene = -1; lastStage = -1; screen.scrollTop = 0; screen.scrollLeft = 0; }
    render(); raf = requestAnimationFrame(tick);
  }
  function sync() {
    const on = inView && !reading && !motion.matches && !document.hidden && !pane.hidden && document.documentElement.dataset.mode !== 'agent';
    if (!on) { pause(); return; }
    if (!playing) { playing = true; previous = null; render(); raf = requestAnimationFrame(tick); }
  }
  // Keyboard users can pause to inspect the demonstration without playback chrome.
  pane.addEventListener('focusin', () => { reading = true; sync(); });
  pane.addEventListener('focusout', () => { reading = false; sync(); });
  document.addEventListener('visibilitychange', sync);
  function accessibility() {
    pane.setAttribute('tabindex', motion.matches ? '-1' : '0');
    transcript.classList.toggle('sr-only', !motion.matches);
    if (motion.matches) { time = steps[2].outputAt; lastScene = -1; }
    sync(); render();
    if (motion.matches) screen.scrollTop = screen.scrollHeight;
  }
  motion.addEventListener('change', accessibility);
  if (typeof MutationObserver !== 'undefined') {
    const visibility = new MutationObserver(sync);
    visibility.observe(pane, {attributes:true, attributeFilter:['hidden']});
    visibility.observe(document.documentElement, {attributes:true, attributeFilter:['data-mode']});
  }
  pane.classList.add('demo-enhanced');
  accessibility(); render();
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(entries => { inView = entries.some(entry => entry.isIntersecting); sync(); }, {threshold:0.15}).observe(pane);
  } else { inView = true; sync(); }
})();
