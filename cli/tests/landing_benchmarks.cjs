const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {quality} = require('../../landing/benchmarks.js');
assert.deepEqual(Object.keys(quality).sort(), ['10', '30']);
assert.equal(quality[30].empty, 77.6);
const elements=JSON.parse(fs.readFileSync(0,'utf8')), nodes=new Map();
class Element {
 constructor(attrs={}) {this.attrs={...attrs};this.hidden='hidden' in attrs;this.value=attrs.value;this.events={};
  this.classList=new Set();this.style={setProperty:(key,value)=>{this[key]=value;}};}
 addEventListener(k,fn){this.events[k]=fn;} fire(k,e={}){this.events[k](e);}
}
for(const attrs of elements) if(attrs.id) nodes.set(attrs.id,new Element(attrs));
const controls=new Map(['.quality-controls','.comparison-board'].map(c=>[c,new Element({hidden:''})]));
nodes.get('benchmarks').querySelector=s=>controls.get(s);
const rate=nodes.get('defect-rate');rate.value='10';
const cliBefore=JSON.stringify(nodes.get('tab-cli').attrs);
vm.runInNewContext(fs.readFileSync('landing/benchmarks.js','utf8'),{document:{getElementById:id=>nodes.get(id)}});
assert.equal(nodes.get('benchmarks').classList.has('is-revealed'),true);
assert.equal(controls.get('.quality-controls').hidden,false);
assert.equal(nodes.get('panel-quality').hidden,false);
assert.equal(nodes.get('empty-score').textContent,'95.4%');
assert.equal(nodes.get('quality-loss').textContent,'4.6');
rate.value='30';rate.fire('change');
assert.equal(nodes.get('empty-score').textContent,'77.6%');
assert.equal(nodes.get('quality-loss').textContent,'22.4');
assert.equal(nodes.get('label-score').textContent,'68.3%');
assert.equal(nodes.get('empty-bar')['--height'],'77.6%');
assert.equal(JSON.stringify(nodes.get('tab-cli').attrs),cliBefore);
// An offscreen board must not consume its entrance; disconnect after it enters.
nodes.get('benchmarks').classList.delete('is-revealed');
let observed, observerCallback, disconnected=false;
class Observer {
 constructor(callback) {observerCallback=callback;}
 observe(element) {observed=element;}
 disconnect() {disconnected=true;}
}
vm.runInNewContext(fs.readFileSync('landing/benchmarks.js','utf8'),{
 document:{getElementById:id=>nodes.get(id)}, IntersectionObserver:Observer,
});
assert.equal(observed,controls.get('.comparison-board'));
observerCallback([{isIntersecting:false}]);
assert.equal(nodes.get('benchmarks').classList.has('is-revealed'),false);
assert.equal(disconnected,false);
observerCallback([{isIntersecting:true}]);
assert.equal(nodes.get('benchmarks').classList.has('is-revealed'),true);
assert.equal(disconnected,true);
console.log('Quality chart, tab isolation and viewport entrance passed');
