const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {timeline,frameAt}=require('../../landing/demo.js');
const recording=JSON.parse(fs.readFileSync('landing/terminal-recording.json','utf8'));
const steps=timeline(recording.commands.map(c=>c.command));
assert.equal(frameAt(-10,steps).stage,0);
assert.equal(frameAt(NaN,steps).typed,0);
assert.equal(frameAt(steps[0].typedAt,steps).typed,1);
assert.equal(frameAt(steps[0].typedAt,steps).output,false);
assert.equal(frameAt(steps[0].outputAt,steps).output,true);
assert.equal(frameAt(steps[1].start,steps).stage,1);
assert.equal(frameAt(Infinity,steps).clear,5);
const attrs=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(a={}) {this.hidden='hidden' in a;this.attrs={...a};this.events={};this.open='open' in a;
  this.dataset={};for(const [k,v] of Object.entries(a))if(k.startsWith('data-'))this.dataset[k.slice(5)]=v;
  this.classes=new Set();this.classList={toggle:(key,on)=>on?this.classes.add(key):this.classes.delete(key)};
  this.scrollHeight=2000;this.scrollTop=0;this.scrollLeft=0;}
 setAttribute(k,v){this.attrs[k]=v;}addEventListener(k,f){this.events[k]=f;}fire(k){this.events[k]();}focus(){this.fire('focusin');}
}
function setup(reduced=false) {
 const nodes=new Map(attrs.filter(a=>a.id).map(a=>[a.id,new Element(a)]));
 const commands=attrs.filter(a=>'data-command' in a).map(a=>new Element(a));
 const runs=commands.map(command=>{const element=new Element(),output=new Element();
  output.textContent='Original recorded output';
  element.querySelector=s=>s==='[data-command]'?command:output;
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
const d=setup();
assert.equal(d.screen.hidden,false);assert.equal(d.nodes.get('terminal-static').classes.has('sr-only'),true);
assert.equal(d.frames.size,0);assert.equal(d.runs[0].output.hidden,true);
d.observe([{isIntersecting:true}]);d.tick(0);d.tick(steps[0].outputAt);
assert.equal(d.runs[0].command.textContent,recording.commands[0].command);
assert.equal(d.runs[0].output.hidden,false);assert.equal(d.runs[1].element.hidden,true);
assert.equal(d.screen.scrollTop,2000);
d.screen.fire('focusin');assert.equal(d.frames.size,0);
d.screen.scrollTop=123;d.observe([{isIntersecting:false}]);d.observe([{isIntersecting:true}]);
assert.equal(d.frames.size,0);assert.equal(d.screen.scrollTop,123,'Reading must preserve scrollback');
d.screen.fire('focusout');assert.equal(d.frames.size,1,'Leaving focus resumes autonomously');
d.tick(0);d.tick(steps[1].outputAt-steps[0].outputAt);
assert.equal(d.runs[0].output.hidden,false);assert.equal(d.runs[1].output.hidden,false);
assert.equal(d.runs[2].element.hidden,true);
d.tick(steps.at(-1).end-steps[0].outputAt+400);
assert.equal(d.nodes.get('terminal-clear').textContent,'clear');
assert.equal(d.nodes.get('terminal-idle').hidden,false);
assert.equal(d.runs.every(r=>!r.element.hidden&&!r.output.hidden),true);
assert.equal(d.runs.every(r=>r.output.textContent==='Original recorded output'),true);
d.tick(steps.at(-1).end-steps[0].outputAt+851);
assert.equal(d.runs[0].output.hidden,true,'A completed cycle restarts without a button');
assert.equal(d.runs[1].element.hidden,true);assert.equal(d.frames.size,1);
d.observe([{isIntersecting:false}]);assert.equal(d.frames.size,0);
d.observe([{isIntersecting:true}]);assert.equal(d.frames.size,1);
d.nodes.get('pane-cli').hidden=true;d.mutate();assert.equal(d.frames.size,0);
d.nodes.get('pane-cli').hidden=false;d.mutate();assert.equal(d.frames.size,1);
d.doc.hidden=true;d.events.visibilitychange();assert.equal(d.frames.size,0);
d.doc.hidden=false;d.events.visibilitychange();assert.equal(d.frames.size,1);
const r=setup(true);r.observe([{isIntersecting:true}]);assert.equal(r.frames.size,0);
assert.equal(r.screen.hidden,true);assert.equal(r.nodes.get('terminal-static').classes.has('sr-only'),false);
r.motion.matches=false;r.motion.change();assert.equal(r.frames.size,1);
r.motion.matches=true;r.motion.change();assert.equal(r.frames.size,0);
assert.equal(r.nodes.get('terminal-static').classes.has('sr-only'),false);
console.log('Automatic terminal cycles, output preservation, reading pause and reduced motion passed');
