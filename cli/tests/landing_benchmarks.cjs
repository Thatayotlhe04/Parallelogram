const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {costs, money} = require('../../landing/benchmarks.js');
// Independently calculated token costs; efficient APIs must not be omitted.
assert.deepEqual(costs('flagship', 1e6, 6000).api.map(m => m.cost), [25000,20000,5000]);
assert.deepEqual(costs('efficient', 1e6, 6000).api.map(m => m.cost), [250,250,675]);
assert.equal(costs('flagship',1e5,6000).perThousand,60);
assert.equal(costs('flagship',5e6,6000).perThousand,1.2);
assert.equal(costs('flagship',1e6,6000).axis,30000);
for (const budget of [0,-1,NaN,Infinity,1000001]) assert.equal(costs('flagship',1e6,budget),null);
for (const volume of [0,-1,Infinity,NaN]) assert.equal(costs('flagship',volume,6000),null);
assert.equal(costs('invalid',1e6,6000),null);
assert.equal(costs('constructor',1e6,6000),null);
assert.equal(money(67.5),'$67.5');
assert.equal(money(6000),'$6K');
const elements=JSON.parse(fs.readFileSync(0,'utf8')), nodes=new Map();
class Element {
 constructor(attrs={}) {this.attrs={...attrs};this.hidden='hidden' in attrs;this.value=attrs.value;
  this.tabIndex=Number(attrs.tabindex||0);this.events={};this.dataset={volume:attrs['data-volume']};
  this.style={setProperty:(key,value)=>{this[key]=value;}};}
 setAttribute(k,v){this.attrs[k]=v;} getAttribute(k){return this.attrs[k];}
 addEventListener(k,fn){this.events[k]=fn;} focus(){this.focused=true;}
 fire(k,e={}){this.events[k](e);}
}
for(const attrs of elements) if(attrs.id) nodes.set(attrs.id,new Element(attrs));
const tabs=['tab-cost','tab-fit','tab-quality'].map(id=>nodes.get(id));
const tablist=new Element({hidden:''});tablist.querySelectorAll=()=>tabs;
const controls=new Map(['.chart-mode','.scenario-input','.workload-options','.quality-controls'].map(c=>[c,new Element({hidden:''})]));
const volumeButtons=elements.filter(e=>'data-volume' in e).map(e=>new Element(e));
nodes.get('benchmarks').querySelector=s=>s==='[role="tablist"]'?tablist:controls.get(s);
nodes.get('benchmarks').querySelectorAll=()=>volumeButtons;
const tier=nodes.get('api-tier'),rate=nodes.get('defect-rate'),budget=nodes.get('owned-budget');
tier.value='flagship';rate.value='10';budget.value='6000';
const cliBefore=JSON.stringify(nodes.get('tab-cli').attrs);
vm.runInNewContext(fs.readFileSync('landing/benchmarks.js','utf8'),{document:{getElementById:id=>nodes.get(id)}});
assert.equal(nodes.get('panel-cost').hidden,false);
assert.equal(nodes.get('panel-fit').hidden,true);
assert.equal(nodes.get('panel-quality').hidden,true);
assert.equal(nodes.get('empty-score').textContent,'95.4%');
assert.equal(nodes.get('cost-value-openai').textContent,'$25K');
tier.value='efficient';tier.fire('change');
assert.equal(nodes.get('cost-name-openai').textContent,'GPT-6 Luna');
assert.equal(nodes.get('cost-value-openai').textContent,'$250');
assert.equal(nodes.get('cost-value-gemini').textContent,'$675');
volumeButtons[0].fire('click');
assert.equal(nodes.get('cost-value-openai').textContent,'$25');
assert.match(nodes.get('own-unit-cost').innerHTML,/60.00/);
assert.equal(volumeButtons[0].getAttribute('aria-pressed'),'true');
budget.value='';budget.fire('input');
assert.equal(budget.getAttribute('aria-invalid'),'true');
assert.equal(nodes.get('budget-error').hidden,false);
assert.equal(nodes.get('cost-value-meta').textContent,'$6K');
budget.value='1000';budget.fire('input');
assert.equal(nodes.get('budget-error').hidden,true);
assert.equal(nodes.get('cost-value-meta').textContent,'$1K');
tier.value='flagship';tier.fire('change');volumeButtons[2].fire('click');
assert.match(nodes.get('cost-insight').textContent,/below all three/);
rate.value='30';rate.fire('change');
assert.equal(nodes.get('empty-score').textContent,'77.6%');
assert.equal(nodes.get('label-score').textContent,'68.3%');
assert.equal(nodes.get('empty-bar')['--height'],'77.6%');
for(const [from,key,to] of [[0,'ArrowLeft',2],[2,'ArrowRight',0],[0,'End',2],[2,'Home',0]]) {
 let prevented=false;tabs[from].fire('keydown',{key,preventDefault:()=>{prevented=true;}});
 assert.equal(prevented,true);assert.equal(tabs[to].getAttribute('aria-selected'),'true');assert.equal(tabs[to].focused,true);
}
tabs[1].fire('click');assert.equal(nodes.get('panel-fit').hidden,false);
assert.equal(JSON.stringify(nodes.get('tab-cli').attrs),cliBefore);
console.log('Cost arithmetic, tier/volume/budget controls, invalid input, quality chart and tab isolation passed');
