// Leila: standalone charts use saved metrics; unavailable results are never zero-filled.
const TrainingCharts = (() => {
  const arms = [['clean_reference','Clean reference','#64748b'],['before_cleaning','Before cleaning','#d88910'],['after_cleaning','After cleaning','#315ac1']];
  const valid = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
  function model(data, macro, skipBefore) {
    const rows = arms.filter(([k]) => data.runs[k] && !(k === 'before_cleaning' && skipBefore));
    const backdoor = rows.some(([k]) => data.runs[k].backdoor_metrics != null);
    return {rows, panels:[{title:'Clean test accuracy',direction:'Higher is better',values:rows.map(([k])=>(data.benchmark?.summary[k]?.accuracy?.mean ?? data.runs[k].metrics?.accuracy))},
      {title:backdoor?'Backdoor ASR':'Macro F1',direction:backdoor?'Lower is better':'Higher is better',values:rows.map(([k])=>backdoor?(data.benchmark?.summary[k]?.asr_non_target?.mean ?? data.runs[k].backdoor_metrics?.asr_non_target):(data.benchmark?.summary[k]?.macro_f1?.mean ?? macro(data.runs[k].metrics)?.f1))}]};
  }
  function filename(data, context={}) {
    const safe = value => String(value || 'unknown').toLowerCase().replace(/[^a-z0-9_-]+/g,'-');
    const rate = Number.isFinite(context.poisonPercent) ? Number(context.poisonPercent.toFixed(2))+'pct' : 'rate-unknown';
    return `poisonguard-${safe(context.dataset || data.dataset)}-${safe(context.attack)}-${rate}-${data.benchmark?"3seeds":"1seed"}-training.png`;
  }
  function render(data, macro, skipBefore, context={}) {
    const canvas=document.getElementById('comparison-chart');if (!canvas) return;
    const ctx=canvas.getContext('2d');if(!ctx) return;
    const {rows,panels}=model(data,macro,skipBefore);
    ctx.fillStyle='#ffffff';ctx.fillRect(0,0,1200,540);
    ctx.fillStyle='#24344b';ctx.font='bold 25px Arial';ctx.fillText('PoisonGuard · Training comparison',32,42);
    panels.forEach((panel,i)=>{
      const left=70+i*590,top=150,width=480,height=270;
      ctx.fillStyle='#24344b';ctx.font='bold 22px Arial';ctx.fillText(panel.title,left,88);
      ctx.font='16px Arial';ctx.fillStyle='#52647c';ctx.fillText(panel.direction,left,115);
      for(let tick=0;tick<=100;tick+=25){const y=top+height-height*tick/100;
        ctx.strokeStyle='#e2e8f0';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+width,y);ctx.stroke();
        ctx.fillStyle='#52647c';ctx.font='13px Arial';ctx.textAlign='right';ctx.fillText(tick+'%',left-8,y+4);ctx.textAlign='left';}
      rows.forEach(([key,title,color],j)=>{
        const value=panel.values[j],slot=width/rows.length,x=left+slot*j+slot/2;
        ctx.fillStyle=color;
        if(valid(value)) ctx.fillRect(x-40,top+height-height*value,80,height*value);
        ctx.textAlign='center';ctx.fillStyle='#24344b';ctx.font='bold 17px Arial';
        ctx.fillText(valid(value)?(value*100).toFixed(2)+'%':'N/A',x,valid(value)?top+height-height*value-10:top+height-12);
        ctx.font='15px Arial';ctx.fillText(title,x,top+height+30);ctx.textAlign='left';
      });
    });
    ctx.fillStyle='#52647c';ctx.font='14px Arial';ctx.fillText(data.benchmark?'Mean of 3 training seeds · Standard deviations in results table · Both axes 0–100%':'Saved run results · Both axes show 0–100% · N/A means unavailable',32,511);
    const baseline=rows.findIndex(([k])=>k===(skipBefore?'clean_reference':'before_cleaning'));
    const after=rows.findIndex(([k])=>k==='after_cleaning');
    const changes=[];
    if(baseline>=0 && after>=0) panels.forEach(p=>{const a=p.values[after],b=p.values[baseline];if(valid(a)&&valid(b)){
      const d=(a-b)*100;changes.push(`${p.title} ${d===0?'unchanged':(d>0?'increased':'decreased')+' by '+Math.abs(d).toFixed(2)+' percentage points'}`);
    }});
    document.getElementById('chart-summary').textContent=changes.length?changes.join('; ')+` compared with ${skipBefore?'clean reference':'before cleaning'}.`:'';
    document.getElementById('result-charts').hidden=false;
    document.getElementById('download-chart').onclick=()=>{
      const status=document.getElementById('chart-download-status');status.textContent='';
      try{canvas.toBlob(blob=>{if(!blob){status.textContent='Could not create the chart image.';return;}
        const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename(data,context);
        document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      },'image/png');}catch(error){status.textContent='Could not download the chart: '+error.message;}
    };
  }
  return {model,render,filename};
})();
