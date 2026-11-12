const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {timeline,frameAt,DURATION,SCENES}=require('../../landing/demo.js');
const recording=JSON.parse(fs.readFileSync('landing/terminal-recording.json','utf8'));
const steps=timeline(recording.commands.map(c=>c.command));
assert.equal(frameAt(-10,steps).scene,0);
assert.equal(frameAt(NaN,steps).typed,0);
assert.equal(frameAt(steps[0].typedAt,steps).output,false);
assert.equal(frameAt(steps[0].outputAt,steps).scene,1);
assert.equal(frameAt(7700,steps).scene,2);
assert.equal(frameAt(steps[1].start,steps).scene,3);
assert.equal(frameAt(15800,steps).scene,4);
assert.equal(frameAt(steps[2].outputAt,steps).scene,5);
assert.equal(frameAt(DURATION-650,steps).reset,true);
assert.ok(15800 > steps[1].outputAt,'Never open the clean file before the fix finishes');
const attrs=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(a={}) {this.hidden='hidden' in a;this.attrs={...a};this.events={};this.dataset={};
  for(const [k,v] of Object.entries(a))if(k.startsWith('data-'))this.dataset[k.slice(5)]=v;
  this.classes=new Set();this.classList={add:key=>this.classes.add(key),toggle:(key,on)=>on?this.classes.add(key):this.classes.delete(key)};
  this.scrollHeight=2000;this.scrollTop=0;this.scrollLeft=0;}
 setAttribute(k,v){this.attrs[k]=v;}addEventListener(k,f){this.events[k]=f;}fire(k){this.events[k]();}
}
function setup(reduced=false) {
 const nodes=new Map(attrs.filter(a=>a.id).map(a=>[a.id,new Element(a)]));
 const commands=attrs.filter(a=>'data-command' in a).map(a=>new Element(a));
 const runs=commands.map(command=>{const element=new Element(),output=new Element();
  output.textContent='Original recorded output';element.querySelector=s=>s==='[data-command]'?command:output;
  return {element,command,output};});
 const screen=nodes.get('terminal-screen');screen.querySelectorAll=()=>runs.map(r=>r.element);
 const events={};const doc={getElementById:id=>nodes.get(id),documentElement:new Element(),hidden:false,addEventListener:(k,f)=>{events[k]=f;}};
 const motion={matches:reduced,addEventListener:(k,f)=>{motion.change=f;}};
 let observe,mutate,sequence=0;const frames=new Map();
 vm.runInNewContext(fs.readFileSync('landing/demo.js','utf8'),{document:doc,window:{matchMedia:()=>motion},
  requestAnimationFrame:f=>{frames.set(++sequence,f);return sequence;},cancelAnimationFrame:id=>frames.delete(id),
  IntersectionObserver:class{constructor(f){observe=f;}observe(){}},MutationObserver:class{constructor(f){mutate=f;}observe(){}},
 });
 const tick=now=>{const pending=[...frames.values()];frames.clear();pending.forEach(cb=>cb(now));};
 return {nodes,runs,screen,doc,events,motion,frames,tick,observe,mutate};
}
const d=setup();const scene=()=>d.nodes.get('demo-desktop').dataset.scene;
assert.equal(d.nodes.get('terminal-static').classes.has('sr-only'),true);
assert.equal(d.frames.size,0);assert.equal(d.runs[0].output.hidden,true);
d.observe([{isIntersecting:true}]);d.tick(0);d.tick(steps[0].outputAt);
assert.equal(scene(),'duplicate');
assert.equal(d.runs[0].command.textContent,recording.commands[0].command);
assert.equal(d.runs[0].output.hidden,false);assert.equal(d.runs[1].element.hidden,true);
d.tick(7700);assert.equal(scene(),'private');
assert.equal(d.nodes.get('story-title').textContent,SCENES[2][2]);
assert.equal(d.nodes.get('source-before').hidden,false);
d.tick(steps[1].start);assert.equal(scene(),'fix');
assert.equal(d.nodes.get('source-after').hidden,true);
d.tick(15800);assert.equal(scene(),'clean');
assert.equal(d.nodes.get('source-before').hidden,true);
assert.equal(d.nodes.get('source-after').hidden,false);
assert.equal(d.nodes.get('editor-filename').textContent,'clean.jsonl');
d.tick(steps[2].outputAt);assert.equal(scene(),'done');
assert.equal(d.runs.every(r=>!r.element.hidden&&!r.output.hidden),true);
assert.equal(d.runs.every(r=>r.output.textContent==='Original recorded output'),true);
d.tick(DURATION+1);assert.equal(scene(),'intro');
assert.equal(d.nodes.get('source-before').hidden,false);
assert.equal(d.nodes.get('source-after').hidden,true);
assert.equal(d.runs[0].output.hidden,true,'The full workflow restarts without a button');
assert.equal(d.frames.size,1);
d.nodes.get('pane-cli').fire('focusin');assert.equal(d.frames.size,0);
d.screen.scrollTop=123;d.observe([{isIntersecting:false}]);d.observe([{isIntersecting:true}]);
assert.equal(d.frames.size,0);assert.equal(d.screen.scrollTop,123);
d.nodes.get('pane-cli').fire('focusout');assert.equal(d.frames.size,1);
d.observe([{isIntersecting:false}]);assert.equal(d.frames.size,0);
d.observe([{isIntersecting:true}]);assert.equal(d.frames.size,1);
d.nodes.get('pane-cli').hidden=true;d.mutate();assert.equal(d.frames.size,0);
d.nodes.get('pane-cli').hidden=false;d.mutate();assert.equal(d.frames.size,1);
d.doc.hidden=true;d.events.visibilitychange();assert.equal(d.frames.size,0);
d.doc.hidden=false;d.events.visibilitychange();assert.equal(d.frames.size,1);
const r=setup(true);r.observe([{isIntersecting:true}]);assert.equal(r.frames.size,0);
assert.equal(r.nodes.get('demo-desktop').dataset.scene,'done');
assert.equal(r.nodes.get('source-after').hidden,false);
assert.equal(r.nodes.get('terminal-static').classes.has('sr-only'),false);
r.motion.matches=false;r.motion.change();assert.equal(r.frames.size,1);
r.motion.matches=true;r.motion.change();assert.equal(r.frames.size,0);
assert.equal(r.nodes.get('terminal-static').classes.has('sr-only'),false);
console.log('File/command synchronization, autonomous cycles, visibility and reduced motion passed');
