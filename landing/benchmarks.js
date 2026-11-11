/* Sourced price snapshot, local scenario calculator and progressive chart UI. */
(() => {
  'use strict';
  const models = {
    flagship: [
      {id:'openai', name:'GPT-6 Astra', input:10, output:50},
      {id:'claude', name:'Claude Fable 5.2', input:8, output:40},
      {id:'gemini', name:'Gemini 4 Carbon', input:2, output:10},
    ],
    efficient: [
      {id:'openai', name:'GPT-6 Luna', input:.1, output:.5},
      {id:'claude', name:'Claude Haiku 5.5', input:.1, output:.5},
      {id:'gemini', name:'Gemini 3.1 Flash-Lite', input:.25, output:1.5},
    ],
  };
  function costs(tier, volume, budget) {
    if (!Object.hasOwn(models, tier) || !Number.isFinite(volume) || volume <= 0 ||
        !Number.isFinite(budget) || budget < 100 || budget > 1000000) return null;
    const api = models[tier].map(model => ({...model,
      cost: volume * (1500 * model.input + 200 * model.output) / 1e6}));
    const max = Math.max(budget, ...api.map(model => model.cost));
    // Three equal grid intervals, with readable rounded tick values.
    const magnitude = 10 ** Math.floor(Math.log10(max / 3));
    const normalized = max / 3 / magnitude;
    const step = [1,2,5,10].find(value => value >= normalized) * magnitude;
    return {api, budget, axis: step * 3, perThousand: budget / volume * 1000};
  }
  function money(value) {
    return '$' + (value >= 1000 ? (value / 1000).toLocaleString('en-US', {maximumFractionDigits:2}) + 'K'
      : value.toLocaleString('en-US', {maximumFractionDigits:2}));
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {costs, money};
  if (typeof document === 'undefined') return;
  const section = document.getElementById('benchmarks');
  if (!section) return;
  const byId = id => document.getElementById(id);
  const tablist = section.querySelector('[role="tablist"]');
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  function select(tab, focus = false) {
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      byId(item.getAttribute('aria-controls')).hidden = !selected;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      const next = {ArrowRight:(index+1)%tabs.length, ArrowLeft:(index+tabs.length-1)%tabs.length,
        Home:0, End:tabs.length-1}[event.key];
      if (next === undefined) return;
      event.preventDefault(); select(tabs[next], true);
    });
  });
  select(tabs[0]); tablist.hidden = false;
  let volume = 1000000;
  const tier = byId('api-tier'), budget = byId('owned-budget');
  const volumeButtons = Array.from(section.querySelectorAll('[data-volume]'));
  function updateCosts() {
    const result = costs(tier.value, volume, Number(budget.value));
    budget.setAttribute('aria-invalid', String(!result));
    byId('budget-error').hidden = Boolean(result);
    if (!result) {
      byId('cost-announcement').textContent = 'Enter a monthly ownership budget between $100 and $1,000,000. The chart retains the last valid estimate.';
      return;
    }
    [...result.api, {id:'meta',cost:result.budget}].forEach(model => {
      byId(`cost-value-${model.id}`).textContent = money(model.cost);
      byId(`cost-bar-${model.id}`).style.setProperty('--height', `${model.cost / result.axis * 100}%`);
      if (model.name) byId(`cost-name-${model.id}`).textContent = model.name;
    });
    [0,1,2,3].forEach(i => {byId(`cost-axis-${i}`).textContent = money(result.axis*i/3);});
    byId('own-unit-cost').innerHTML = `$${result.perThousand.toFixed(2)}<span> / 1K tasks</span>`;
    const volumeText = volume === 100000 ? '100 thousand' : volume / 1e6 + ' million';
    byId('workload-label').textContent = `${volumeText} tasks`;
    const cheapest = Math.min(...result.api.map(model => model.cost));
    byId('cost-insight').textContent = result.budget <= cheapest
      ? `Your budget is below all three API estimates at this volume. Confirm task quality and serving capacity before switching.`
      : `The lowest listed API estimate is ${money(cheapest)}/month at this volume. Ownership is not automatically cheaper.`;
    byId('cost-announcement').textContent = `${volumeText} tasks monthly. ${result.api.map(m => `${m.name}: ${money(m.cost)}`).join('; ')}; your fine-tune: ${money(result.budget)}. Cost scenario, not a quality ranking.`;
  }
  tier.addEventListener('change', updateCosts);
  budget.addEventListener('input', updateCosts);
  volumeButtons.forEach(button => button.addEventListener('click', () => {
    volume = Number(button.dataset.volume);
    volumeButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    updateCosts();
  }));
  ['.chart-mode','.scenario-input','.workload-options','.quality-controls'].forEach(selector => {
    section.querySelector(selector).hidden = false;
  });
  updateCosts();
  const quality = {10:{empty:95.4,label:87.8,affected:12.1},30:{empty:77.6,label:68.33333333333333,affected:31.9}};
  const rate = byId('defect-rate');
  function updateQuality() {
    const result = quality[rate.value]; if (!result) return;
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
