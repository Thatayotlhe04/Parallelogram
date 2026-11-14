/* The "why check first" chart: switch between the two injection rates. */
(() => {
  'use strict';
  const quality = {10:{empty:95.4,label:87.8,affected:12.1},30:{empty:77.6,label:68.33333333333333,affected:31.9}};
  if (typeof module !== 'undefined' && module.exports) module.exports = {quality};
  if (typeof document === 'undefined') return;
  const section = document.getElementById('benchmarks');
  if (!section) return;
  const byId = id => document.getElementById(id);
  section.querySelector('.quality-controls').hidden = false;
  const rate = byId('defect-rate');
  function updateQuality() {
    const result = quality[rate.value]; if (!result) return;
    byId('quality-loss').textContent = (100 - result.empty).toFixed(1);
    ['empty','label'].forEach(key => {
      byId(`${key}-score`).textContent = `${result[key].toFixed(1)}%`;
      byId(`${key}-bar`).style.setProperty('--height', `${result[key]}%`);
    });
    byId('quality-context').textContent = `${rate.value}% injection setting; systematic label errors affected ${result.affected}% of rows. 383K parameters · 5 seeds · greedy decoding.`;
    byId('quality-announcement').textContent = `Clean data 100%; empty answers ${result.empty.toFixed(1)}%; wrong labels ${result.label.toFixed(1)}%.`;
  }
  rate.addEventListener('change', updateQuality); updateQuality();
  // Start the one-time entrance when the board reaches the viewport. Nothing
  // is hidden while waiting, so charts remain readable if observation fails.
  if (typeof IntersectionObserver !== 'undefined') {
    const entrance = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      section.classList.add('is-revealed');
      entrance.disconnect();
    }, {threshold: 0.08});
    entrance.observe(section.querySelector('.comparison-board'));
  } else {
    section.classList.add('is-revealed');
  }
})();
