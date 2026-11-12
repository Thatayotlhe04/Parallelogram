const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {choose,COMPACT,STAGE_WIDTH}=require('../../landing/demo.js');
assert.deepEqual(choose(STAGE_WIDTH,false),{layout:'wide',size:'wide',scale:1,play:true});
assert.equal(choose(1184,false).scale,1184/STAGE_WIDTH);
assert.deepEqual(choose(COMPACT-1,false),{layout:'compact',size:'narrow',scale:1,play:true});
assert.equal(choose(1184,true).play,false);

const attrs=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(a={}) {this.hidden='hidden' in a;this.attrs={...a};this.events={};this.dataset={};this.style={};this.clientWidth=1184;
  for(const [k,v] of Object.entries(a))if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-(\w)/g,(_,c)=>c.toUpperCase())]=v;
  this.classes=new Set((a.class||'').split(/\s+/).filter(Boolean));
  this.classList={add:k=>this.classes.add(k),toggle:(k,on)=>on?this.classes.add(k):this.classes.delete(k)};}
 setAttribute(k,v){this.attrs[k]=v;}addEventListener(k,f){this.events[k]=f;}fire(k){this.events[k]();}
}
function setup(reduced=false,width=1184) {
 const nodes=new Map(attrs.filter(a=>a.id).map(a=>[a.id,new Element(a)]));
 nodes.get('demo-desktop').clientWidth=width;
 const video=nodes.get('demo-video');video.paused=true;video.plays=0;
 video.play=()=>{video.paused=false;video.plays++;return Promise.resolve();};video.pause=()=>{video.paused=true;};
 const events={};const doc={getElementById:id=>nodes.get(id),documentElement:new Element(),hidden:false,addEventListener:(k,f)=>{events[k]=f;}};
 const motion={matches:reduced,addEventListener:(k,f)=>{motion.change=f;}};
 let observe,mutate;
 vm.runInNewContext(fs.readFileSync('landing/demo.js','utf8'),{document:doc,window:{matchMedia:()=>motion,innerWidth:width},
  IntersectionObserver:class{constructor(f){observe=f;}observe(){}},MutationObserver:class{constructor(f){mutate=f;}observe(){}},
  ResizeObserver:class{observe(){}},
 });
 return {nodes,video,doc,events,motion,observe:e=>observe(e),mutate:()=>mutate(),pane:nodes.get('pane-cli')};
}
const d=setup();
assert.equal(d.video.src,'demo/terminal-108.mp4');
assert.equal(d.video.poster,'demo/terminal-108.webp');
assert.equal(d.nodes.get('demo-desktop').dataset.layout,'wide');
assert.equal(d.nodes.get('scene-stage').style.transform,`scale(${1184/STAGE_WIDTH})`);
assert.equal(d.nodes.get('terminal-static').classes.has('sr-only'),true);
assert.equal(d.pane.classes.has('demo-enhanced'),true);
assert.equal(d.video.paused,true,'Nothing plays before the panel is on screen');
d.observe([{isIntersecting:true}]);assert.equal(d.video.paused,false);
d.observe([{isIntersecting:false}]);assert.equal(d.video.paused,true);
d.observe([{isIntersecting:true}]);
d.pane.fire('focusin');assert.equal(d.video.paused,true,'Keyboard focus pauses for reading');
d.pane.fire('focusout');assert.equal(d.video.paused,false);
d.pane.hidden=true;d.mutate();assert.equal(d.video.paused,true);
d.pane.hidden=false;d.mutate();assert.equal(d.video.paused,false);
d.doc.hidden=true;d.events.visibilitychange();assert.equal(d.video.paused,true);
d.doc.hidden=false;d.events.visibilitychange();assert.equal(d.video.paused,false);
d.doc.documentElement.dataset.mode='agent';d.mutate();assert.equal(d.video.paused,true);
// Narrow screens: the 60-column recording alone.
const m=setup(false,390);
assert.equal(m.video.src,'demo/terminal-60.mp4');
assert.equal(m.nodes.get('demo-desktop').dataset.layout,'compact');
assert.equal(m.nodes.get('scene-stage').style.transform,'none');
// Reduced motion: the finished run as a still, and the full transcript.
const r=setup(true);r.observe([{isIntersecting:true}]);
assert.equal(r.video.plays,0);assert.equal(r.video.poster,'demo/terminal-108.webp');
assert.equal(r.nodes.get('terminal-static').classes.has('sr-only'),false);
r.motion.matches=false;r.motion.change();assert.equal(r.video.paused,false);
r.motion.matches=true;r.motion.change();assert.equal(r.video.paused,true);
console.log('Layout, visibility, focus, narrow recording and reduced motion passed');
