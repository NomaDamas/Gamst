/* Shared dependency-free observability renderer. Model labels are always text. */
(function(root){
  const colors=['#516bd1','#45aade','#6cc581','#e7c869','#eb817f','#b681dc','#64b8b0','#a5a5b5'];
  const colorFor=label=>colors[Array.from(label).reduce((hash,char)=>(hash*31+char.charCodeAt(0))>>>0,0)%colors.length];
  const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
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
      const t=(ko,en)=>this.locale==='en'?en:ko;
      const groups=[];
      for(const [name,o] of Object.entries(observations)){
        if(name.startsWith('classification.')&&o.value?.labels)groups.push([name,o.value.labels,o,'model']);
      }
      // A measured expression profile is available even without a classifier.
      const face=observations['scene.faces'];
      if(face?.value?.[0]?.blendshapes)groups.push([t('표정 움직임','Facial movement'),face.value[0].blendshapes,face,'measurement']);
      const nodes=[],names=[],labelOptions=new Map();
      for(const [name,scores,obs,kind] of groups.slice(0,8)){
        const all=Object.entries(scores).filter(([k,v])=>typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=1).sort((a,b)=>b[1]-a[1]);
        if(!all.length)continue;
        const choice=this.labels.get(name)||'*',selectedLabel=all.some(([label])=>label===choice)?choice:'*';labelOptions.set(name,[['*',t('전체','All')],...all.map(([key])=>[key,key]).sort((a,b)=>a[0].localeCompare(b[0]))]);
        const ranked=all.slice(0,8),visible=selectedLabel==='*'?ranked:all.filter(([label])=>label===selectedLabel),total=all.reduce((s,x)=>s+x[1],0),section=el('section');
        section.style.cssText='padding:16px 0;border-bottom:1px solid #e7e7ed';
        const title=el('div',name);title.style.cssText='font-weight:650;font-size:13px;margin-bottom:12px';section.append(title);
        const summary=el('div');summary.style.cssText='display:flex;gap:20px;align-items:center;flex-wrap:wrap';
        const donut=el('div');donut.style.cssText='width:100px;height:100px;border-radius:50%;position:relative;flex-shrink:0';
        let edge=0;const stops=[];
        ranked.forEach(([label,value],i)=>{const next=edge+(total?value/total*100:0);stops.push(`${colorFor(label)} ${edge}% ${next}%`);edge=next});
        stops.push(`#e7e7ed ${edge}% 100%`);donut.style.background=`conic-gradient(${stops.join(',')})`;
        const center=el('div',live&&obs.fresh!==false?(total?Math.round(ranked[0][1]/total*100)+'%':'—'):'—');center.style.cssText='position:absolute;inset:12px;border-radius:50%;background:white;display:grid;place-items:center;font-size:20px;font-weight:650';donut.append(center);donut.setAttribute('aria-label',name+' '+t('상대 점수 비중','relative score share'));
        const info=el('div');info.append(el('strong',ranked[0][0]),el('div',`${t('최대 점수','Top score')} ${ranked[0][1].toFixed(3)} · ${kind==='model'?t('모델 출력','model output'):t('관측값','measurement')}`),el('small',t('점수 합계 기준 비중 · ','Share of score total · ')+(live&&obs.fresh!==false?'LIVE':'STOPPED / STALE')));summary.append(donut,info);section.append(summary);
        const history=this.history.get(name)||[];
        if(live&&this.last.get(name)!==obs.timestamp_ms){history.push({time:obs.timestamp_ms,scores:{...scores}});if(history.length>60)history.shift();this.history.set(name,history);this.last.set(name,obs.timestamp_ms)}
        const first=history[0]?.time||0,last=history.at(-1)?.time||first,span=Math.max(1,last-first);
        const graph=document.createElementNS('http://www.w3.org/2000/svg','svg');graph.setAttribute('viewBox','0 0 300 60');graph.setAttribute('role','img');graph.setAttribute('aria-label',name+' '+t('시계열','time series'));graph.style.cssText='width:100%;height:60px;margin:12px 0;background:#fafafd;border-radius:8px';
        visible.forEach(([label,value],i)=>{
          const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('fill','none');line.setAttribute('stroke',colorFor(label));line.setAttribute('stroke-width','1.5');line.setAttribute('points',history.filter(p=>Number.isFinite(p.scores[label])).map(p=>`${(p.time-first)/span*300},${58-p.scores[label]*56}`).join(' '));graph.append(line);
          const row=el('div');row.style.cssText='display:grid;grid-template-columns:minmax(90px,1fr) 2fr 48px;gap:10px;align-items:center;margin-top:7px;font-size:11px';const track=el('div'),bar=el('div');track.style.cssText='height:5px;background:#f0f0f5;border-radius:9px';bar.style.cssText=`height:5px;border-radius:9px;background:${colorFor(label)};width:${value*100}%`;track.append(bar);row.append(el('span',label),track,el('span',value.toFixed(3)));section.append(row)
        });section.append(graph,el('small',`${(span/1000).toFixed(1)}s · ${history.length} samples`));nodes.push(section);names.push(name)
      }
      // Arbitrary scalar sensors retain their own units rather than becoming percentages.
      const scalars=Object.entries(observations).filter(([name,o])=>typeof o.value==='number'&&Number.isFinite(o.value)).slice(0,12);
      for(const [name,o] of scalars){const key='scalar:'+name,h=this.history.get(key)||[];if(live&&this.last.get(key)!==o.timestamp_ms){h.push({time:o.timestamp_ms,value:o.value});if(h.length>60)h.shift();this.history.set(key,h);this.last.set(key,o.timestamp_ms)}const section=el('section');section.style.cssText='padding:12px 0;border-bottom:1px solid #e7e7ed';section.append(el('strong',name+' · '+o.value.toFixed(3)));const min=Math.min(...h.map(p=>p.value),o.value),max=Math.max(...h.map(p=>p.value),o.value),span=Math.max(1,(h.at(-1)?.time||0)-(h[0]?.time||0));const graph=document.createElementNS('http://www.w3.org/2000/svg','svg'),line=document.createElementNS('http://www.w3.org/2000/svg','polyline');graph.setAttribute('viewBox','0 0 300 40');graph.style.cssText='display:block;width:100%;height:40px';line.setAttribute('fill','none');line.setAttribute('stroke','#516bd1');line.setAttribute('stroke-width','1.5');line.setAttribute('points',h.map(p=>`${(p.time-h[0].time)/span*300},${max===min?20:38-(p.value-min)/(max-min)*36}`).join(' '));graph.append(line);section.append(graph,el('small',`min ${min.toFixed(3)} · max ${max.toFixed(3)}`));nodes.push(section);names.push(key)}
      if(!nodes.length){this.history.clear();this.last.clear();this.nav.hidden=true;this.detailNav.hidden=true;this.plot.replaceChildren(el('div',t('관측 / 분류 모델 입력 대기','Waiting for observations / classifier output')));return}
      const activeKeys=new Set([...groups.map(g=>g[0]),...scalars.map(([n])=>'scalar:'+n)]);
      for(const key of this.history.keys())if(!activeKeys.has(key)){this.history.delete(key);this.last.delete(key)}
      if(!names.includes(this.selected))this.selected=names[0];
      this.nav.hidden=false;
      this.tabs(this.nav,names.map(name=>[name,name.replace(/^scalar:/,'')]),this.selected,key=>{this.selected=key;this.render(this.observations,this.live)});
      const options=labelOptions.get(this.selected)||[];this.detailNav.hidden=!options.length;
      if(options.length){let selected=this.labels.get(this.selected)||'*';if(!options.some(([key])=>key===selected)){selected='*';this.labels.set(this.selected,selected)}this.tabs(this.detailNav,options,selected,key=>{this.labels.set(this.selected,key);this.render(this.observations,this.live)})}
      const node=nodes[names.indexOf(this.selected)];node.setAttribute('role','tabpanel');node.setAttribute('aria-label',this.selected);this.plot.replaceChildren(node)
    }
  }
  root.ContextCharts=ContextCharts;
})(globalThis);
