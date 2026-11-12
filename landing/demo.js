/* Continuous replay of captured CLI stdout. The browser runs no shell commands. */
(() => {
  'use strict';
  function timeline(commands) {
    let at = 0;
    return commands.map((command, index) => {
      const start = at;
      const typedAt = start + command.length * 24 + 240;
      const outputAt = typedAt + 300;
      at = outputAt + (index === 0 ? 6500 : index === 1 ? 4800 : 4200);
      return {start, typedAt, outputAt, end: at};
    });
  }
  function frameAt(time, steps) {
    const duration = steps[steps.length - 1].end;
    const elapsed = Math.max(0, Number(time) || 0);
    const index = steps.findIndex(step => elapsed < step.end);
    const stage = index < 0 ? steps.length - 1 : index;
    const step = steps[stage];
    return {stage, typed: Math.min(1, (elapsed - step.start) / (step.typedAt - step.start)),
      output: elapsed >= step.outputAt,
      clear: Math.max(0, Math.min(5, Math.floor((elapsed - duration) / 65)))};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {timeline, frameAt};
  if (typeof document === 'undefined') return;
  const byId = id => document.getElementById(id);
  const screen = byId('terminal-screen');
  if (!screen) return;
  const pane = byId('pane-cli'), transcript = byId('terminal-static');
  const runs = Array.from(screen.querySelectorAll('[data-run]')).map(element => ({
    element, command: element.querySelector('[data-command]'), output: element.querySelector('[data-output]'),
  }));
  const steps = timeline(runs.map(run => run.command.dataset.command));
  const duration = steps[steps.length - 1].end;
  const cycle = duration + 850;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let time = 0, playing = false, inView = false, reading = false, previous = null, raf = 0;
  let lastStage = -1, lastOutput = false;
  function render() {
    const state = frameAt(time, steps);
    screen.classList.toggle('is-running', playing);
    runs.forEach((run, index) => {
      run.element.hidden = index > state.stage;
      const full = run.command.dataset.command;
      const text = index < state.stage ? full : index === state.stage ? full.slice(0, Math.ceil(full.length * state.typed)) : '';
      if (run.command.textContent !== text) run.command.textContent = text;
      run.output.hidden = !(index < state.stage || index === state.stage && state.output);
      run.element.classList.toggle('is-typing', index === state.stage && !state.output);
    });
    byId('terminal-idle').hidden = state.stage !== runs.length - 1 || !state.output;
    byId('terminal-clear').textContent = 'clear'.slice(0, state.clear);
    // Scroll only this terminal when new output arrives. Page position stays fixed.
    if (playing && (state.stage !== lastStage || state.output !== lastOutput)) screen.scrollTop = screen.scrollHeight;
    lastStage = state.stage; lastOutput = state.output;
  }
  function pause() {
    playing = false; previous = null; cancelAnimationFrame(raf); render();
  }
  function tick(now) {
    if (!playing) return;
    if (previous !== null) time += now - previous;
    previous = now;
    if (time >= cycle) {
      time %= cycle; lastStage = -1; lastOutput = false;
      screen.scrollTop = 0; screen.scrollLeft = 0;
    }
    render(); raf = requestAnimationFrame(tick);
  }
  function sync() {
    const on = inView && !reading && !motion.matches && !document.hidden && !pane.hidden && document.documentElement.dataset.mode !== 'agent';
    if (!on) { pause(); return; }
    if (playing) return;
    playing = true; previous = null; render(); raf = requestAnimationFrame(tick);
  }
  screen.addEventListener('focusin', () => { reading = true; sync(); });
  screen.addEventListener('focusout', () => { reading = false; sync(); });
  screen.addEventListener('wheel', () => { screen.focus(); }, {passive:true});
  document.addEventListener('visibilitychange', sync);
  function accessibility() {
    screen.hidden = motion.matches;
    transcript.classList.toggle('sr-only', !motion.matches);
    sync();
  }
  motion.addEventListener('change', accessibility);
  if (typeof MutationObserver !== 'undefined') {
    const visibility = new MutationObserver(sync);
    visibility.observe(pane, {attributes:true, attributeFilter:['hidden']});
    visibility.observe(document.documentElement, {attributes:true, attributeFilter:['data-mode']});
  }
  accessibility(); render();
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(entries => {
      inView = entries.some(entry => entry.isIntersecting); sync();
    }, {threshold:0.3}).observe(pane);
  } else { inView = true; sync(); }
})();
