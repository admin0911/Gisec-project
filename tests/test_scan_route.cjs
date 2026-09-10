// Leila: exercise scan selection without starting detectors or changing saved results.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(path.join(__dirname,'..','frontend','scan-results.js'),'utf8');
function page(search='',entry=null,saved=null) {
  const calls=[],nodes={},storage=new Map(saved ? [['label-flip-job',saved]] : []);
  const history={state:entry,replaceState(state,_,url){this.state=state;this.url=url}};
  vm.runInNewContext(code,{
    document:{getElementById:id=>nodes[id]||(nodes[id]={})},location:{search},history,URLSearchParams,
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
    window:{addEventListener(){}},fetch:async url=>{calls.push(url);return {ok:false}},setTimeout
  });
  return {calls,history,storage};
}
let p=page('?job=chosen',{poisonGuardScan:{job:'older'}},'latest');
assert.equal(p.calls[0],'/api/jobs/chosen');assert.equal(p.history.url,'/scan');
assert.equal(p.storage.get('label-flip-job'),'chosen');
p=page('',p.history.state,'another-tab');
assert.equal(p.calls[0],'/api/jobs/chosen');
p=page('',null,'saved');assert.equal(p.calls[0],'/api/jobs/saved');
p=page();assert.equal(p.calls.length,0);assert.equal(p.history.url,'/scan');
console.log('Passed: explicit scan links, refresh/history selection, storage fallback and empty context.');
