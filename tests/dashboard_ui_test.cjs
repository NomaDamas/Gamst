const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
 constructor(){this.textContent='';this.style={setProperty(k,v){this[k]=v}};this.children=[];this.value='';this.checked=false;this.parentElement={setAttribute(){},removeAttribute(){}};this.classList={add(){},remove(){}}}
 getAttribute(k){return this[k]} showModal(){this.open=true} close(){this.open=false;this.onclose?.()}
 focus(){this.focused=true} getBoundingClientRect(){return {width:1000,left:0}} setPointerCapture(){} hasPointerCapture(){return true} releasePointerCapture(){}
 append(...els){this.children.push(...els)} replaceChildren(...els){this.children=els} setAttribute(k,v){this[k]=v}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const html=fs.readFileSync('humansense/dashboard.html','utf8');
const context=vm.createContext({document:{body:new Element(),getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element()},window:{addEventListener(){}},AbortController,fetch:()=>new Promise(()=>{}),setTimeout,clearInterval});
assert.equal(fs.readFileSync('humansense/observability.js','utf8'),fs.readFileSync('site/observability.js','utf8'));
vm.runInContext(fs.readFileSync('humansense/i18n.js','utf8'),context);
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
const section=context.testCharts.plot.children[0],summary=section.children[1];
assert.equal(summary.children[0].children[0].textContent,'60%');assert.equal(summary.children[1].children[0].textContent,'joy');
vm.runInContext('testCharts.clear()',context);assert.equal(context.testCharts.history.size,0);
console.log('Category charts: proportions, dominant label, bounded history and shared public asset passed');

vm.runInContext("selectTab('vlm')",context);assert.equal(get('panel-vlm').hidden,false);assert.equal(get('panel-context').hidden,true);assert.equal(get('tab-vlm')['aria-selected'],'true');
get('tab-vlm').onkeydown({key:'ArrowRight',preventDefault(){}});assert.equal(get('tab-observability')['aria-selected'],'true');
vm.runInContext('setCameraRatio(90)',context);assert.equal(get('splitter')['aria-valuenow'],'65');
get('splitter').onpointerdown({button:0,pointerId:1,preventDefault(){}});get('splitter').onpointermove({clientX:400});get('splitter').onpointerup({pointerId:1});assert.equal(get('splitter')['aria-valuenow'],'40');
vm.runInContext('setCaptureStatus(true)',context);assert.equal(get('statusPill')['data-active'],'true');assert.equal(get('status').textContent,'Observing');vm.runInContext('setCaptureStatus(false)',context);assert.equal(get('statusPill')['data-active'],'false');
vm.runInContext('testCharts.render(categories)',context);const buttons=context.testCharts.tabCache.get(context.testCharts.detailNav).buttons;buttons[1].onclick();const oldHistory=context.testCharts.history.get('classification.emotion_fixture').length;vm.runInContext('testCharts.render(categories)',context);assert.equal(context.testCharts.history.get('classification.emotion_fixture').length,oldHistory);assert.equal(context.testCharts.tabCache.get(context.testCharts.detailNav).buttons[1],buttons[1]);assert.equal(buttons[1]['aria-selected'],'true');
console.log('Main/detail tabs, pointer resizing, rainbow states and preserved history passed');

assert.equal(context.observationTitle('scene.person_count','ko'),'감지된 사람 수');
get('video').videoWidth=640;get('video').videoHeight=360;get('faceGuide').checked=true;get('personGuide').checked=true;
context.overlayFixture={state:{paused:false},context:{observations:{'scene.faces':{fresh:true,value:[{landmarks:[{x:.5,y:.25}]}]},'scene.person_tracks':{fresh:true,value:[{visible:true,bbox:{x:.1,y:.1,width:.4,height:.8},track_id:'camera:person-1',association_quality:'matched'}]}}}};
vm.runInContext('drawGuides(overlayFixture)',context);assert.equal(get('cameraOverlay').children.length,3);assert.equal(get('cameraOverlay').children[0].cx,'320');
get('faceGuide').checked=false;vm.runInContext('drawGuides(overlayFixture)',context);assert.equal(get('cameraOverlay').children.length,2);
context.overlayFixture.state.paused=true;vm.runInContext('drawGuides(overlayFixture)',context);assert.equal(get('cameraOverlay').children.length,0);
console.log('Friendly labels and camera overlay visibility/coordinates passed');

assert.equal(context.CymI18n.text('설정','en'),'Settings');assert.equal(context.CymI18n.text('설정','zh'),'设置');assert.equal(context.CymI18n.text('VLM connection test','ko'),'VLM 연결 테스트');

assert(!html.includes('id="liveDot"'));assert(!html.includes('id="tab-settings"'));assert(!html.includes('type="checkbox"'));
get('openSettings').onclick();assert.equal(get('settingsDialog').open,true);get('closeSettings').onclick();assert.equal(get('settingsDialog').open,false);assert.equal(get('openSettings').focused,true);
get('consent').onclick();assert.equal(get('consent').checked,true);assert.equal(get('consent')['aria-pressed'],'true');get('consent').onclick();assert.equal(get('consent').checked,false);
get('useVideo').disabled=true;const before=get('useVideo').checked;get('useVideo').onclick();assert.equal(get('useVideo').checked,before);
assert(html.indexOf('id="scene"')<html.indexOf('id="vlmOptions"'));
console.log('Modal settings, semantic toggle buttons and compact VLM layout passed');
