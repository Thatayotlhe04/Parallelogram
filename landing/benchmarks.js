/* Progressive enhancement for the research section. All results and sources
   remain readable without JavaScript; no requests, storage or dependencies. */
(() => {
  'use strict';
  const section = document.getElementById('benchmarks');
  if (!section) return;
  const tablist = section.querySelector('[role="tablist"]');
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  function select(tab, focus = false) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0, End: tabs.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      select(tabs[next], true);
    });
  });
  select(tabs[0]);
  tablist.hidden = false;

  // E2 mean exact-match accuracy, greedy decoding, five paired seeds.
  // Source revision and intervals are linked in the adjacent methodology.
  const results = {
    10: {empty: 95.4, label: 87.8, affected: 12.1},
    30: {empty: 77.6, label: 68.33333333333333, affected: 31.9},
  };
  const rate = document.getElementById('defect-rate');
  function update() {
    const result = results[rate.value];
    if (!result) return;
    ['empty', 'label'].forEach(key => {
      document.getElementById(`${key}-score`).textContent = `${result[key].toFixed(1)}%`;
      document.getElementById(`${key}-bar`).style.setProperty('--bar', `${result[key]}%`);
    });
    document.getElementById('quality-context').textContent = `${rate.value}% injection setting; systematic errors affected ${result.affected}% of rows. Five seeds · 383K-parameter model · greedy decoding.`;
    document.getElementById('quality-announcement').textContent = `At the ${rate.value}% setting: clean data 100%; empty answers ${result.empty.toFixed(1)}%; systematic label errors ${result.label.toFixed(1)}%.`;
  }
  rate.addEventListener('change', update);
  section.querySelector('.bench-controls').hidden = false;
  // A browser may restore the select on reload.
  update();
})();
