/* Shared dependency-free observability renderer. Model labels are always text. */
(function(root){
  const colors=['#516bd1','#45aade','#6cc581','#e7c869','#eb817f','#b681dc','#64b8b0','#a5a5b5'];
  const colorFor=label=>colors[Array.from(label).reduce((hash,char)=>(hash*31+char.charCodeAt(0))>>>0,0)%colors.length];
  const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
  const namesKo={'scene.person_count':'감지된 사람 수','scene.person_boxes':'사람 영역','scene.person_tracks':'사람 움직임 추적','audio.rms':'목소리 크기','audio.dbfs':'음량 · dBFS','audio.energy_active':'소리 감지','speech.activity':'발화 여부','speech.probability':'발화 점수','speech.transcribed_word_count':'인식된 단어 수','speech.transcribed_words_per_minute':'분당 인식 단어 수','classification.voice_emotion':'목소리 감정','scene.faces':'얼굴 · 표정','scene.description':'주변 장면','scene.window':'장면 관찰 구간','text.transcript':'인식된 말'};
  const namesEn={'scene.person_count':'People detected','scene.person_boxes':'Person regions','scene.person_tracks':'Person tracks','audio.rms':'Voice level','audio.dbfs':'Level · dBFS','speech.activity':'Speech activity','speech.probability':'Speech score','speech.transcribed_word_count':'Transcribed words','speech.transcribed_words_per_minute':'Words per minute','classification.voice_emotion':'Voice emotion'};
  const namesZh={'scene.person_count':'检测到的人数','scene.pose_count':'身体姿态数量','scene.person_boxes':'人物区域','scene.person_tracks':'人物跟踪','scene.body_tracks':'身体跟踪','scene.poses':'身体姿态','audio.rms':'声音强度','audio.dbfs':'音量 · dBFS','speech.activity':'语音活动','speech.probability':'语音分数','speech.transcribed_word_count':'已识别词数','speech.transcribed_words_per_minute':'每分钟词数','classification.voice_emotion':'语音情绪','scene.faces':'面部动作'};
  Object.assign(namesEn,{'scene.pose_count':'Body poses detected','scene.poses':'Body joints','scene.body_tracks':'Body tracks'});Object.assign(namesKo,{'scene.pose_count':'감지된 몸 자세 수','scene.poses':'몸 관절','scene.body_tracks':'몸 자세 추적'});
  root.observationTitle=(name,locale=root.CYMLocale||'ko')=>(locale==='en'?namesEn:locale==='zh'?namesZh:namesKo)[name]||name.replace(/^classification\./,'').replace(/[_.]/g,' ');
  const faceTitles={_neutral:'기본 표정',browDown:'눈썹 내림',browInnerUp:'눈썹 안쪽 올림',browOuterUp:'눈썹 바깥 올림',cheekPuff:'볼 부풀림',cheekSquint:'볼 올림',eyeBlink:'눈 깜빡임',eyeLookDown:'아래 보기',eyeLookIn:'안쪽 보기',eyeLookOut:'바깥 보기',eyeLookUp:'위 보기',eyeSquint:'눈 가늘게 뜸',eyeWide:'눈 크게 뜸',jawForward:'턱 내밀기',jawLeft:'턱 왼쪽',jawOpen:'입 벌림',jawRight:'턱 오른쪽',mouthClose:'입 다물기',mouthDimple:'입꼬리 당김',mouthFrown:'입꼬리 내림',mouthFunnel:'입 모으기',mouthLeft:'입 왼쪽',mouthLowerDown:'아랫입술 내림',mouthPress:'입술 누름',mouthPucker:'입술 내밀기',mouthRight:'입 오른쪽',mouthRollLower:'아랫입술 말기',mouthRollUpper:'윗입술 말기',mouthShrugLower:'아랫입술 올림',mouthShrugUpper:'윗입술 올림',mouthSmile:'미소',mouthStretch:'입 가로 늘림',mouthUpperUp:'윗입술 올림',noseSneer:'코 찡그림',tongueOut:'혀 내밀기'};
  root.faceMovementTitle=label=>{if(faceTitles[label])return faceTitles[label];const base=label.replace(/(Left|Right)$/,'');return faceTitles[base]?faceTitles[base]+(label.endsWith('Left')?' · 왼쪽':' · 오른쪽'):label};
  class ContextCharts {
    constructor(mount,locale='ko'){this.locale=locale;this.mount=mount;this.history=new Map();this.last=new Map();this.selected=null;this.labels=new Map();this.nav=el('div');this.detailNav=el('div');this.plot=el('div');this.tabCache=new Map();this.mount.replaceChildren(this.nav,this.detailNav,this.plot)}
    tabs(host,items,selected,choose){
      host.setAttribute('role','tablist');host.setAttribute('aria-label',host===this.nav?'관측 항목':'시계열 카테고리');host.style.cssText='display:flex;gap:6px;overflow-x:auto;padding:12px 0;scrollbar-width:thin';
      const signature=JSON.stringify(items);let cached=this.tabCache.get(host);
      if(!cached||cached.signature!==signature){const buttons=items.map(([key,label])=>{const button=el('button',label);button.type='button';button.setAttribute('role','tab');button.style.cssText='white-space:nowrap;border:1px solid #e7e7ed;border-radius:20px;padding:7px 11px;font-size:11px;cursor:pointer';return button});host.replaceChildren(...buttons);cached={signature,buttons};this.tabCache.set(host,cached)}
      cached.buttons.forEach((button,i)=>{const key=items[i][0],active=key===selected;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;button.style.background=active?'#eef0fb':'#fff';button.style.color=active?'#516bd1':'#747480';button.onclick=()=>choose(key);button.onkeydown=e=>{let n;if(e.key==='ArrowRight')n=(i+1)%items.length;else if(e.key==='ArrowLeft')n=(i+items.length-1)%items.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=items.length-1;else return;e.preventDefault();choose(items[n][0]);this.tabCache.get(host).buttons[n].focus()}})
    }
    clear(){this.history.clear();this.last.clear();this.selected=null;this.labels.clear();this.nav.replaceChildren();this.detailNav.replaceChildren();this.plot.replaceChildren();this.tabCache.clear()}
    render(observations,live=true){
      this.observations=observations;this.live=live;
      const t=(ko,en)=>this.locale==='en'?en:this.locale==='zh'?({'몸 움직임':'身体动作','전체':'全部','표정 움직임':'面部动作','상대 점수 비중':'相对分数占比','최대 점수':'最高分','모델 출력':'模型输出','관측값':'观察值','점수 합계 기준 비중 · ':'总分占比 · ','시계열':'时间线','관측 / 분류 모델 입력 대기':'等待观察或分类输出','중립':'中性','기쁨':'高兴','분노':'愤怒','슬픔':'悲伤'}[ko]||en):ko;
      const groups=[];
      for(const [name,o] of Object.entries(observations)){
        if(name.startsWith('classification.')&&o.value?.labels)groups.push([name,o.value.labels,o,'model']);
      }
      // A measured expression profile is available even without a classifier.
      const face=observations['scene.faces'];
      for(const [i,f] of (face?.value||[]).entries())if(f.blendshapes)groups.push([t('표정 움직임','Facial movement')+' · '+(i+1),f.blendshapes,face,'measurement']);
      const body=observations['scene.poses'];
      for(const pose of body?.value||[]){if(!pose.track_id||pose.association_quality==='uncertain')continue;const scores={};for(const [key,value] of Object.entries(pose.geometry||{}))if(typeof value==='boolean')scores[key.replaceAll('_',' ')]=Number(value);if(Object.keys(scores).length)groups.push([t('몸 움직임','Body movement')+' · '+pose.track_id,scores,body,'measurement'])}
      const nodes=[],names=[],labelOptions=new Map();
      for(const [name,scores,obs,kind] of groups.slice(0,16)){
        const all=Object.entries(scores).filter(([k,v])=>typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=1).sort((a,b)=>b[1]-a[1]);
        if(!all.length)continue;
        const labelTitle=label=>name==='classification.voice_emotion'?({neu:t('중립','Neutral'),hap:t('기쁨','Happy'),ang:t('분노','Angry'),sad:t('슬픔','Sad')}[label]||label):(kind==='measurement'&&this.locale==='ko'?root.faceMovementTitle(label):label.replace(/([a-z])([A-Z])/g,'$1 $2'));
        const choice=this.labels.get(name)||'*',selectedLabel=all.some(([label])=>label===choice)?choice:'*';labelOptions.set(name,[['*',t('전체','All')],...all.map(([key])=>[key,labelTitle(key)]).sort((a,b)=>a[0].localeCompare(b[0]))]);
        const ranked=all.slice(0,8),visible=selectedLabel==='*'?ranked:all.filter(([label])=>label===selectedLabel),total=all.reduce((s,x)=>s+x[1],0),section=el('section');
        section.style.cssText='padding:16px 0;border-bottom:1px solid #e7e7ed';
        const title=el('div',root.observationTitle(name,this.locale));title.style.cssText='font-weight:650;font-size:13px;margin-bottom:12px';section.append(title);
        const summary=el('div');summary.style.cssText='display:flex;gap:20px;align-items:center;flex-wrap:wrap';
        const donut=el('div');donut.style.cssText='width:100px;height:100px;border-radius:50%;position:relative;flex-shrink:0';
        let edge=0;const stops=[];
        ranked.forEach(([label,value],i)=>{const next=edge+(total?value/total*100:0);stops.push(`${colorFor(label)} ${edge}% ${next}%`);edge=next});
        stops.push(`#e7e7ed ${edge}% 100%`);donut.style.background=`conic-gradient(${stops.join(',')})`;
        const center=el('div',live&&obs.fresh!==false?(total?Math.round(ranked[0][1]/total*100)+'%':'—'):'—');center.style.cssText='position:absolute;inset:12px;border-radius:50%;background:white;display:grid;place-items:center;font-size:20px;font-weight:650';donut.append(center);donut.setAttribute('aria-label',name+' '+t('상대 점수 비중','relative score share'));
        const info=el('div');info.append(el('strong',labelTitle(ranked[0][0])),el('div',`${t('최대 점수','Top score')} ${ranked[0][1].toFixed(3)} · ${kind==='model'?t('모델 출력','model output'):t('관측값','measurement')}`),el('small',t('점수 합계 기준 비중 · ','Share of score total · ')+(live&&obs.fresh!==false?'LIVE':'STOPPED / STALE')));summary.append(donut,info);section.append(summary);
        const history=this.history.get(name)||[];
        if(live&&this.last.get(name)!==obs.timestamp_ms){history.push({time:obs.timestamp_ms,scores:{...scores}});if(history.length>60)history.shift();this.history.set(name,history);this.last.set(name,obs.timestamp_ms)}
        const first=history[0]?.time||0,last=history.at(-1)?.time||first,span=Math.max(1,last-first);
        const graph=document.createElementNS('http://www.w3.org/2000/svg','svg');graph.setAttribute('viewBox','0 0 300 60');graph.setAttribute('role','img');graph.setAttribute('aria-label',name+' '+t('시계열','time series'));graph.style.cssText='width:100%;height:60px;margin:12px 0;background:#fafafd;border-radius:8px';
        visible.forEach(([label,value],i)=>{
          const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('fill','none');line.setAttribute('stroke',colorFor(label));line.setAttribute('stroke-width','1.5');line.setAttribute('points',history.filter(p=>Number.isFinite(p.scores[label])).map(p=>`${(p.time-first)/span*300},${58-p.scores[label]*56}`).join(' '));graph.append(line);
          const row=el('div');row.style.cssText='display:grid;grid-template-columns:minmax(90px,1fr) 2fr 48px;gap:10px;align-items:center;margin-top:7px;font-size:11px';const track=el('div'),bar=el('div');track.style.cssText='height:5px;background:#f0f0f5;border-radius:9px';bar.style.cssText=`height:5px;border-radius:9px;background:${colorFor(label)};width:${value*100}%`;track.append(bar);row.append(el('span',labelTitle(label)),track,el('span',value.toFixed(3)));section.append(row)
        });section.append(graph,el('small',`${(span/1000).toFixed(1)}s · ${history.length} samples`));nodes.push(section);names.push(name)
      }
      // Arbitrary scalar sensors retain their own units rather than becoming percentages.
      const scalars=Object.entries(observations).filter(([name,o])=>typeof o.value==='number'&&Number.isFinite(o.value)).slice(0,12);
      for(const [name,o] of scalars){const key='scalar:'+name,h=this.history.get(key)||[];if(live&&this.last.get(key)!==o.timestamp_ms){h.push({time:o.timestamp_ms,value:o.value});if(h.length>60)h.shift();this.history.set(key,h);this.last.set(key,o.timestamp_ms)}const section=el('section');section.style.cssText='padding:12px 0;border-bottom:1px solid #e7e7ed';section.append(el('strong',root.observationTitle(name,this.locale)+' · '+o.value.toFixed(3)));const min=Math.min(...h.map(p=>p.value),o.value),max=Math.max(...h.map(p=>p.value),o.value),span=Math.max(1,(h.at(-1)?.time||0)-(h[0]?.time||0));const graph=document.createElementNS('http://www.w3.org/2000/svg','svg'),line=document.createElementNS('http://www.w3.org/2000/svg','polyline');graph.setAttribute('viewBox','0 0 300 40');graph.style.cssText='display:block;width:100%;height:40px';line.setAttribute('fill','none');line.setAttribute('stroke','#516bd1');line.setAttribute('stroke-width','1.5');line.setAttribute('points',h.map(p=>`${(p.time-h[0].time)/span*300},${max===min?20:38-(p.value-min)/(max-min)*36}`).join(' '));graph.append(line);section.append(graph,el('small',`min ${min.toFixed(3)} · max ${max.toFixed(3)}`));nodes.push(section);names.push(key)}
      if(!nodes.length){this.history.clear();this.last.clear();this.nav.hidden=true;this.detailNav.hidden=true;this.plot.replaceChildren(el('div',t('관측 / 분류 모델 입력 대기','Waiting for observations / classifier output')));return}
      const activeKeys=new Set([...groups.map(g=>g[0]),...scalars.map(([n])=>'scalar:'+n)]);
      for(const key of this.history.keys())if(!activeKeys.has(key)){this.history.delete(key);this.last.delete(key)}
      if(!names.includes(this.selected))this.selected=names[0];
      this.nav.hidden=false;
      this.tabs(this.nav,names.map(name=>[name,root.observationTitle(name.replace(/^scalar:/,''),this.locale)]),this.selected,key=>{this.selected=key;this.render(this.observations,this.live)});
      const options=labelOptions.get(this.selected)||[];this.detailNav.hidden=!options.length;
      if(options.length){let selected=this.labels.get(this.selected)||'*';if(!options.some(([key])=>key===selected)){selected='*';this.labels.set(this.selected,selected)}this.tabs(this.detailNav,options,selected,key=>{this.labels.set(this.selected,key);this.render(this.observations,this.live)})}
      const node=nodes[names.indexOf(this.selected)];node.setAttribute('role','tabpanel');node.setAttribute('aria-label',this.selected);this.plot.replaceChildren(node)
    }
  }
  root.ContextCharts=ContextCharts;
})(globalThis);
