// Exercise the shipped script against IDs and attributes parsed from the real
// page. The small DOM fixture keeps these tests independent of a browser install.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = new Map();
class Element {
  constructor(attrs = {}) {
    this.attrs = {...attrs}; this.hidden = 'hidden' in attrs;
    this.tabIndex = Number(attrs.tabindex || 0); this.events = {};
    this.style = {setProperty: (key, value) => {this[key] = value;}};
  }
  setAttribute(key, value) {this.attrs[key] = value;}
  getAttribute(key) {return this.attrs[key];}
  addEventListener(key, fn) {this.events[key] = fn;}
  focus() {this.focused = true;}
  fire(key, event = {}) {this.events[key](event);}
}
for (const attrs of elements) if (attrs.id) nodes.set(attrs.id, new Element(attrs));
const tabs = ['tab-quality', 'tab-ownership'].map(id => nodes.get(id));
const tablist = new Element({hidden: ''});
tablist.querySelectorAll = () => tabs;
const controls = new Element({hidden: ''});
nodes.get('benchmarks').querySelector = selector => selector === '[role="tablist"]' ? tablist : controls;
const rate = nodes.get('defect-rate');
rate.value = '10'; // Simulate browser-restored form state on reload.
const cliTab = nodes.get('tab-cli');
const beforeCLI = JSON.stringify(cliTab.attrs);
vm.runInNewContext(fs.readFileSync('landing/benchmarks.js', 'utf8'), {
  document: {getElementById: id => nodes.get(id)},
});
assert.equal(tablist.hidden, false);
assert.equal(controls.hidden, false);
assert.equal(nodes.get('panel-quality').hidden, false);
assert.equal(nodes.get('panel-ownership').hidden, true);
assert.equal(nodes.get('empty-score').textContent, '95.4%');
assert.equal(nodes.get('label-score').textContent, '87.8%');
rate.value = '30'; rate.fire('change');
assert.equal(nodes.get('empty-score').textContent, '77.6%');
assert.equal(nodes.get('label-score').textContent, '68.3%');
assert.equal(nodes.get('empty-bar')['--bar'], '77.6%');
assert.match(nodes.get('quality-context').textContent, /31.9%/);
assert.match(nodes.get('quality-announcement').textContent, /77.6%/);
tabs[1].fire('click');
assert.equal(nodes.get('panel-quality').hidden, true);
assert.equal(nodes.get('panel-ownership').hidden, false);
assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
assert.equal(tabs[0].tabIndex, -1);
for (const [from, key, to] of [[1,'ArrowRight',0],[0,'ArrowLeft',1],[1,'Home',0],[0,'End',1]]) {
  let prevented = false;
  tabs[from].fire('keydown', {key, preventDefault: () => {prevented = true;}});
  assert.equal(prevented, true);
  assert.equal(tabs[to].getAttribute('aria-selected'), 'true');
  assert.equal(tabs[to].focused, true);
}
assert.equal(JSON.stringify(cliTab.attrs), beforeCLI, 'Research tabs must not alter CLI/CI/Agents');
console.log('Research controls: restored form, chart values, announcements, keyboard and isolation passed');
