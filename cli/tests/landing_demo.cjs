const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {frameAt,duration}=require('../../landing/demo.js');
assert.equal(duration,20000);
assert.deepEqual(frameAt(-100),{time:0,stage:0,local:0,typed:0,ended:false});
assert.equal(frameAt(7999).stage,0);
assert.equal(frameAt(8000).stage,1);
assert.equal(frameAt(15000).stage,2);
assert.equal(frameAt(16000).typed,.625);
assert.equal(frameAt(999999).ended,true);
assert.equal(frameAt(NaN).time,0);
const attrs=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(a={}) {this.attrs={...a};this.hidden='hidden' in a;this.events={};this.value=a.value||'';this.open='open' in a;
  this.dataset={};for(const [k,v] of Object.entries(a)) if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v;
  this.classes=new Set();this.classList={toggle:(key,on)=>on?this.classes.add(key):this.classes.delete(key)};
  this.style={setProperty:(key,value)=>{this[key]=value;}};}
 setAttribute(k,v){this.attrs[k]=v;} addEventListener(k,fn){this.events[k]=fn;}
 fire(k){this.events[k]();}
}
function setup(reduced=false) {
 const nodes=new Map(attrs.filter(a=>a.id).map(a=>[a.id,new Element(a)]));
 const scenes=[];
 for(const a of attrs) {
  if('data-scene' in a)scenes.push({element:new Element(a),lines:[]});
  if('data-command' in a)scenes.at(-1).command=new Element(a);
  if('data-at' in a)scenes.at(-1).lines.push(new Element(a));
 }
 scenes.forEach(scene=>{scene.element.querySelector=()=>scene.command;scene.element.querySelectorAll=()=>scene.lines;});
 const chapters=attrs.filter(a=>'data-demo-stage' in a).map(a=>new Element(a));
 const player=nodes.get('cli-demo');player.querySelectorAll=s=>s==='[data-scene]'?scenes.map(s=>s.element):chapters;
 const root=new Element(); const events={};
 const doc={getElementById:id=>nodes.get(id),documentElement:root,hidden:false,addEventListener:(k,f)=>{events[k]=f;}};
 const motion={matches:reduced,addEventListener:(k,f)=>{motion.change=f;}};
 let observe,mutate,sequence=0;const frames=new Map();
 vm.runInNewContext(fs.readFileSync('landing/demo.js','utf8'),{
  document:doc,window:{matchMedia:()=>motion},
  requestAnimationFrame:cb=>{frames.set(++sequence,cb);return sequence;},cancelAnimationFrame:id=>frames.delete(id),
  IntersectionObserver:class{constructor(cb){observe=cb;}observe(){}},
  MutationObserver:class{constructor(cb){mutate=cb;}observe(){}},
 });
 const tick=now=>{const pending=[...frames.values()];frames.clear();pending.forEach(cb=>cb(now));};
 return {nodes,scenes,chapters,player,doc,events,motion,frames,tick,observe,mutate};
}
const d=setup();
assert.equal(d.player.hidden,false);
assert.equal(d.nodes.get('demo-transcript').open,false);
assert.equal(d.frames.size,0);
d.observe([{isIntersecting:true}]);d.tick(0);d.tick(6000);
assert.equal(d.scenes[0].lines.every(l=>l.classes.has('is-visible')),true);
assert.equal(d.nodes.get('demo-toggle').attrs['aria-label'],'Pause demo');
d.nodes.get('demo-toggle').fire('click');
assert.equal(d.frames.size,0);
assert.equal(d.nodes.get('demo-seek').value,'6.0');
d.observe([{isIntersecting:false}]);d.observe([{isIntersecting:true}]);
assert.equal(d.frames.size,0,'Returning to view must not override a pause');
d.nodes.get('demo-seek').value='12';d.nodes.get('demo-seek').fire('input');
assert.equal(d.player.dataset.stage,'1');assert.equal(d.nodes.get('demo-seek').value,'12.0');
d.chapters[2].fire('click');
assert.equal(d.player.dataset.stage,'2');assert.equal(d.scenes[2].lines.every(l=>l.classes.has('is-visible')),true);
d.nodes.get('demo-replay').fire('click');d.tick(0);d.tick(8000);
assert.equal(d.player.dataset.stage,'1');d.tick(15000);assert.equal(d.player.dataset.stage,'2');d.tick(20000);
assert.equal(d.frames.size,0);assert.equal(d.nodes.get('demo-clock').textContent,'0:20 / 0:20');
assert.equal(d.nodes.get('demo-toggle').attrs['aria-label'],'Replay demo from start');
d.nodes.get('demo-toggle').fire('click');d.tick(0);d.tick(1000);
assert.equal(d.nodes.get('demo-seek').value,'1.0');
d.nodes.get('pane-cli').hidden=true;d.mutate();assert.equal(d.frames.size,0);
d.nodes.get('pane-cli').hidden=false;d.nodes.get('demo-toggle').fire('click');d.doc.hidden=true;d.events.visibilitychange();
assert.equal(d.frames.size,0);
const r=setup(true);r.observe([{isIntersecting:true}]);
assert.equal(r.frames.size,0,'Reduced motion must not autoplay');
assert.equal(r.scenes[0].command.textContent,r.scenes[0].command.dataset.command);
assert.equal(r.scenes[0].lines.every(l=>l.classes.has('is-visible')),true);
r.nodes.get('demo-toggle').fire('click');r.motion.change();assert.equal(r.frames.size,0);
console.log('Demo timing, pause/resume, seeking, chapters, completion, replay, visibility and reduced motion passed');
