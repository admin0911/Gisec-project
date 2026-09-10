// Leila: short review addresses preserve selection without saving review decisions.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(path.join(__dirname,'..','frontend','human-review.js'),'utf8');
async function page(search='',entry=null,saved=null) {
  const calls=[],nodes={},storage=new Map(saved ? [['poison-guard-review',JSON.stringify(saved)]] : []);
  const history={state:entry,replaceState(state,_,url){this.state=state;this.url=url}};
  vm.runInNewContext(code,{
    document:{getElementById:id=>nodes[id]||(nodes[id]={addEventListener(){},setAttribute(){},replaceChildren(){}}),querySelectorAll:()=>[]},
    location:{search},history,URLSearchParams,
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},window:{addEventListener(){}},
    fetch:async(url,options)=>{calls.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>({revision:0,pages:1,total:0,resolved:0,unsure:0,unreviewed:0,items:[]})}}
  });
  await new Promise(resolve=>setImmediate(resolve));
  return {calls,history,nodes};
}
(async()=>{
  let p=await page('?job=chosen&group=suspected_label_flip',null,{job:'old'});
  assert.equal(p.history.url,'/review');assert.equal(p.calls[0].body.job_id,'chosen');
  assert.equal(p.nodes.group.value,'suspected_label_flip');
  p=await page('',p.history.state,{job:'another-tab',group:'uncertain'});
  assert.equal(p.calls[0].body.job_id,'chosen');assert.equal(p.nodes.group.value,'suspected_label_flip');
  p=await page('',null,{job:'saved',group:'uncertain'});assert.equal(p.calls[0].body.job_id,'saved');
  p=await page();assert.equal(p.calls.length,0);
  console.log('Passed: review links, refresh/history context, storage fallback and empty context.');
})().catch(error=>{console.error(error);process.exitCode=1});
