const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
 constructor(){this.textContent='';this.style={};this.children=[];this.value='';this.checked=false;this.parentElement={setAttribute(){},removeAttribute(){}};this.classList={add(){},remove(){}}}
 append(...els){this.children.push(...els)} replaceChildren(...els){this.children=els} setAttribute(k,v){this[k]=v}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const html=fs.readFileSync('humansense/dashboard.html','utf8');
const context=vm.createContext({document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element()},window:{addEventListener(){}},AbortController,fetch:()=>new Promise(()=>{}),setTimeout,clearInterval});
assert.equal(fs.readFileSync('humansense/observability.js','utf8'),fs.readFileSync('site/observability.js','utf8'));
vm.runInContext(fs.readFileSync('humansense/observability.js','utf8'),context);
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const d={context:{observations:{'audio.rms':{value:.2,fresh:true,age_ms:0,timestamp_ms:100},'scene.faces':{value:[{frame_face_index:0,blendshapes:{mouthSmileLeft:.7}}],fresh:true,age_ms:0,timestamp_ms:100},'scene.description':{value:'<script>untrusted</script>',fresh:true,age_ms:10,timestamp_ms:100}}},state:{signals:{'scene.faces':{observation:{value:[{blendshapes:{mouthSmileLeft:.7,mouthSmileRight:.5,eyeBlinkLeft:.1,eyeBlinkRight:.1,jawOpen:.4,browInnerUp:.2}}]}}}},vlm:{available:true,enabled:true,interval:3,window_seconds:4,sample_fps:2,effective_sample_fps:2,buffered_frames:8,max_frames:8},workers:[],adapter_errors:[]};
context.fixture=d;vm.runInContext('render(fixture)',context);
assert.equal(get('value-smile').textContent,'60');assert.equal(get('value-rms').textContent,'0.200');assert.equal(get('scene').textContent,'<script>untrusted</script>');assert.equal(get('contextRows').children.length,3);
d.context.observations['scene.faces'].fresh=false;vm.runInContext('render(fixture)',context);assert.equal(get('value-smile').textContent,'—');
for(const phrase of ['지금 이 순간을','얼굴 움직임은 감정의 확률','원본은 메모리에서만'])assert(!html.includes(phrase));
console.log('Dashboard rendering: gauges, table, stale values, text escaping passed');

context.categoryMount=new Element();
vm.runInContext("globalThis.testCharts=new ContextCharts(categoryMount)",context);
for(let i=0;i<70;i++){context.categories={'classification.emotion_fixture':{value:{labels:{joy:.6,neutral:.3,sadness:.1}},timestamp_ms:i*250,fresh:true}};vm.runInContext('testCharts.render(categories)',context)}
assert.equal(context.testCharts.history.get('classification.emotion_fixture').length,60);
const section=context.categoryMount.children[0],summary=section.children[1];
assert.equal(summary.children[0].children[0].textContent,'60%');assert.equal(summary.children[1].children[0].textContent,'joy');
vm.runInContext('testCharts.clear()',context);assert.equal(context.testCharts.history.size,0);
console.log('Category charts: proportions, dominant label, bounded history and shared public asset passed');
