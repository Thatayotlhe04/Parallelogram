/* The CLI panel: a real editor view of the dataset beside a screen recording of a real
   terminal session. The stage scales as one picture; below 760px only the 60-column
   recording shows. It plays only while visible and never under reduced motion. */
(() => {
  'use strict';
  const STAGE_WIDTH = 1240;
  const COMPACT = 760;

  function choose(width, reduced) {
    const compact = width < COMPACT;
    return {layout: compact ? 'compact' : 'wide', size: compact ? 'narrow' : 'wide',
      scale: compact ? 1 : width / STAGE_WIDTH, play: !reduced};
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {choose, COMPACT, STAGE_WIDTH};
  if (typeof document === 'undefined') return;
  const pane = document.getElementById('pane-cli');
  const scene = document.getElementById('demo-desktop');
  const stage = document.getElementById('scene-stage');
  const video = document.getElementById('demo-video');
  const transcript = document.getElementById('terminal-static');
  if (!pane || !scene || !video) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let inView = false, reading = false, size = '';

  function layout() {
    const choice = choose(scene.clientWidth || window.innerWidth, motion.matches);
    scene.dataset.layout = choice.layout;
    stage.style.transform = choice.layout === 'compact' ? 'none' : `scale(${choice.scale})`;
    if (choice.size === size) return;
    size = choice.size;
    video.poster = video.dataset[`${size}Poster`];
    video.src = video.dataset[size];
    video.preload = motion.matches ? 'none' : 'auto';
  }
  function sync() {
    const on = inView && !reading && !motion.matches && !document.hidden && !pane.hidden
      && document.documentElement.dataset.mode !== 'agent';
    if (on) {
      const playing = video.play();
      if (playing && playing.catch) playing.catch(() => {});
    } else if (!video.paused) {
      video.pause();
    }
  }
  function accessibility() {
    pane.setAttribute('tabindex', motion.matches ? '-1' : '0');
    transcript.classList.toggle('sr-only', !motion.matches);
    layout();
    sync();
  }
  // Keyboard users can pause the recording to read it; there is no playback chrome.
  pane.addEventListener('focusin', () => { reading = true; sync(); });
  pane.addEventListener('focusout', () => { reading = false; sync(); });
  document.addEventListener('visibilitychange', sync);
  motion.addEventListener('change', accessibility);
  if (typeof MutationObserver !== 'undefined') {
    const visibility = new MutationObserver(sync);
    visibility.observe(pane, {attributes: true, attributeFilter: ['hidden']});
    visibility.observe(document.documentElement, {attributes: true, attributeFilter: ['data-mode']});
  }
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => { layout(); sync(); }).observe(scene);
  pane.classList.add('demo-enhanced');
  accessibility();
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(entries => { inView = entries.some(entry => entry.isIntersecting); sync(); }, {threshold: 0.2}).observe(pane);
  } else { inView = true; sync(); }
})();
