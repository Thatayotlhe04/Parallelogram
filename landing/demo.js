/* A narrated terminal animation, not a live CLI or a timing benchmark. */
(() => {
  'use strict';
  const duration = 20000;
  const starts = [0, 8000, 15000];
  const captions = [
    'Find the problems before training starts.',
    'Repair what is safe. Drop the rest with a reason.',
    'Re-check the output. Let a clean exit gate your next step.',
  ];
  function frameAt(milliseconds) {
    const time = Math.max(0, Math.min(duration, Number(milliseconds) || 0));
    const stage = time >= starts[2] ? 2 : time >= starts[1] ? 1 : 0;
    const local = time - starts[stage];
    return {time, stage, local, typed: Math.min(1, local / 1600), ended: time === duration};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {frameAt, duration};
  if (typeof document === 'undefined') return;
  const player = document.getElementById('cli-demo');
  if (!player) return;
  const byId = id => document.getElementById(id);
  const pane = byId('pane-cli');
  const toggle = byId('demo-toggle'), seek = byId('demo-seek');
  const scenes = Array.from(player.querySelectorAll('[data-scene]')).map(element => ({
    element, command: element.querySelector('[data-command]'),
    lines: Array.from(element.querySelectorAll('[data-at]')),
  }));
  const chapters = Array.from(player.querySelectorAll('[data-demo-stage]'));
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let time = 0, playing = false, started = false, raf = 0, previous = null;
  function render() {
    const frame = frameAt(time);
    player.dataset.stage = String(frame.stage);
    player.classList.toggle('is-playing', playing);
    scenes.forEach((scene, index) => {
      scene.element.hidden = index !== frame.stage;
      if (index !== frame.stage) return;
      const full = scene.command.dataset.command;
      const typed = motion.matches ? 1 : frame.typed;
      scene.command.textContent = full.slice(0, Math.ceil(full.length * typed));
      scene.element.classList.toggle('is-typing', playing && typed < 1);
      scene.lines.forEach(line => {
        line.classList.toggle('is-visible', motion.matches || frame.local >= Number(line.dataset.at));
      });
    });
    chapters.forEach((chapter,index) => {
      chapter.setAttribute('aria-pressed', String(index === frame.stage));
      chapter.classList.toggle('is-complete', index < frame.stage || frame.ended);
    });
    byId('demo-caption').textContent = captions[frame.stage];
    toggle.innerHTML = playing ? '<span aria-hidden="true">Ⅱ</span> Pause' : frame.ended ? '<span aria-hidden="true">↻</span> Replay' : '<span aria-hidden="true">▶</span> Play';
    toggle.setAttribute('aria-label', playing ? 'Pause demo' : frame.ended ? 'Replay demo from start' : 'Play demo');
    seek.value = (time / 1000).toFixed(1);
    seek.style.setProperty('--progress', `${time / duration * 100}%`);
    seek.setAttribute('aria-valuetext', `${Math.floor(time / 1000)} of 20 seconds`);
    byId('demo-clock').textContent = `0:${String(Math.floor(time / 1000)).padStart(2,'0')} / 0:20`;
  }
  function pause() {
    playing = false; previous = null;
    cancelAnimationFrame(raf); render();
  }
  function tick(now) {
    if (!playing) return;
    if (previous !== null) time = Math.min(duration, time + now - previous);
    previous = now;
    if (time === duration) { pause(); return; }
    render(); raf = requestAnimationFrame(tick);
  }
  function play() {
    if (playing) return;
    if (time >= duration) time = 0;
    started = true; playing = true; previous = null;
    render(); raf = requestAnimationFrame(tick);
  }
  toggle.addEventListener('click', () => { started = true; if (playing) pause(); else play(); });
  byId('demo-replay').addEventListener('click', () => { pause(); time = 0; play(); });
  seek.addEventListener('input', () => {
    const next = Number(seek.value) * 1000;
    started = true; pause(); time = frameAt(next).time; render();
  });
  chapters.forEach((chapter, index) => chapter.addEventListener('click', () => {
    started = true; pause();
    // A chapter selection shows its result immediately; Play continues from it.
    time = starts[index] + (index === 0 ? 6000 : index === 1 ? 5500 : 3500);
    render();
  }));
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  motion.addEventListener('change', () => { pause(); });
  // Switching to CI/Agents or Agent view must not leave an invisible demo playing.
  if (typeof MutationObserver !== 'undefined') {
    const visibility = new MutationObserver(() => {
      if (pane.hidden || document.documentElement.dataset.mode === 'agent') pause();
    });
    visibility.observe(pane, {attributes:true, attributeFilter:['hidden']});
    visibility.observe(document.documentElement, {attributes:true, attributeFilter:['data-mode']});
  }
  byId('demo-transcript').open = false;
  player.hidden = false;
  render();
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(entries => {
      const visible = entries.some(entry => entry.isIntersecting);
      if (!visible) pause();
      else if (!started && !motion.matches && !document.hidden && !pane.hidden && document.documentElement.dataset.mode !== 'agent') play();
    }, {threshold:0.35}).observe(player);
  }
})();
