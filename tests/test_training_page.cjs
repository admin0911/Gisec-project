// Exercise the real controller with controlled network responses; no training is started.
const fs = require('fs'), path = require('path'), vm = require('vm'), assert = require('assert');
const root = path.join(__dirname,'..','frontend');
const html = fs.readFileSync(path.join(root,'training.html'),'utf8');
const code = fs.readFileSync(path.join(root,'training.js'),'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function page({poisoned=3, removed=9, caught=2, available=true, statuses=[],search='?version=v&scan=s&run=r',saved=null,entry=null}={}) {
  const nodes = {}, timers = [], requests = [];
  const element = () => ({hidden:true,disabled:false,textContent:'',value:'5',children:[],listeners:{},
    addEventListener(name,fn){this.listeners[name]=fn},replaceChildren(){this.children=[]},appendChild(c){this.children.push(c)}});
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes[match[1]]=element();
  const completed = {version:'v',status:'complete',progress:100,message:'Complete',result:{
    runs:{clean_reference:{metrics:{accuracy:.8}},before_cleaning:{metrics:{accuracy:.6}},after_cleaning:{metrics:{accuracy:.7}}}}};
  const storage = new Map(saved ? [['poison-guard-training',JSON.stringify(saved)]] : []);
  const history = {state:entry,address:null,replaceState(state,_,url){this.state=state;this.address=url}};
  const context = {document:{getElementById:id=>nodes[id],createElement:element},URLSearchParams,
    location:{search},history,sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},setTimeout:fn=>timers.push(fn),
    fetch:async (url,options) => {
      requests.push({url,body:JSON.parse(options.body)});
      let result;
      if (url.endsWith('/prepared')) result={version:'v',scan_id:'s',summary:{kept:20-removed,quarantined:removed,unresolved:0,total:20}};
      else if (url.endsWith('/evaluation')) result={available,reason:'No truth',original:{clean:20-poisoned,poisoned,total:20},
        removed:{clean:removed-caught,poisoned:caught,total:removed},kept:{clean:20-poisoned-removed+caught,poisoned:poisoned-caught,total:20-removed}};
      else if (url.endsWith('/job')) {
        result=statuses.length ? statuses.shift() : completed;
        if (result==='network') throw new TypeError('Failed to fetch');
      } else throw new Error('Unexpected mutation: '+url);
      return {ok:true,status:200,json:async()=>result};
    }};
  vm.runInNewContext(code,context); await flush();
  return {nodes,timers,requests,history,storage};
}

(async () => {
  let p=await page({search:'?version=v&scan=s&dataset=mnist'});
  assert.equal(p.nodes.train.disabled,false);
  assert.equal(p.nodes.prepared.hidden,false);
  assert.equal(p.nodes['evaluation-precision'].textContent,'22.22%');
  assert(p.requests.some(r=>r.url==='/api/training/prepared'));
  assert(p.requests.some(r=>r.url==='/api/training/evaluation'));
  assert.equal(p.history.state.poisonGuardTraining.dataset,'mnist');
  p=await page();
  assert.equal(p.nodes['evaluation-precision'].textContent,'22.22%');
  assert.equal(p.nodes['evaluation-recall'].textContent,'66.67%');
  assert.equal(p.nodes['evaluation-retention'].textContent,'58.82%');
  assert.equal(p.nodes['precision-details'].hidden,false);
  // Leila: legacy runs must show unavailable metrics instead of invented zero scores.
  assert.deepEqual(p.nodes['comparison-rows'].children[0].children.map(c=>c.textContent),
    ['Clean reference','80.00%','N/A','N/A']);
  const comparison = matrix => ({version:'v',status:'complete',progress:100,message:'Complete',result:{
    runs:{before_cleaning:{metrics:{accuracy:.7,confusion_matrix:matrix}},
      after_cleaning:{metrics:{accuracy:.7,confusion_matrix:matrix}}}}});
  // Unequal class errors distinguish macro averaging from overall accuracy.
  p=await page({statuses:[comparison([[8,2],[1,1]])]});
  assert.deepEqual(p.nodes['comparison-rows'].children[1].children.map(c=>c.textContent),
    ['Before cleaning','70.00%','62.11%','61.11%']);
  assert.equal(p.nodes['comparison-rows'].children[0].children[2].textContent,'Not run');
  p=await page({statuses:[comparison([[2,0],[2,0]])]});
  assert.equal(p.nodes['comparison-rows'].children[1].children[2].textContent,'33.33%');
  assert.equal(p.nodes['comparison-rows'].children[1].children[3].textContent,'25.00%');
  for (const matrix of [[[0,0],[0,0]],[[1,-1],[0,2]],[[1,2],[3]]]) {
    p=await page({statuses:[comparison(matrix)]});
    assert.equal(p.nodes['comparison-rows'].children[1].children[2].textContent,'N/A');
  }
  p=await page({poisoned:0,caught:0});
  assert.equal(p.nodes['evaluation-precision'].textContent,'0.00%');
  assert.equal(p.nodes['evaluation-recall'].textContent,'N/A');
  p=await page({poisoned:0,caught:0,removed:0});
  assert.equal(p.nodes['evaluation-precision'].textContent,'N/A');
  assert.equal(p.nodes['evaluation-retention'].textContent,'100.00%');
  p=await page({poisoned:3,caught:0,removed:0});
  assert.equal(p.nodes['evaluation-recall'].textContent,'0.00%');
  p=await page({poisoned:20,caught:9,removed:9});
  assert.equal(p.nodes['evaluation-retention'].textContent,'N/A');
  p=await page({available:false});
  assert.equal(p.nodes['evaluation-metrics'].hidden,true);

  p=await page({statuses:['network']});
  assert(p.nodes['training-status'].textContent.includes('Reconnecting'));
  assert.equal(p.nodes.train.disabled,true);
  p.timers.shift()(); await flush();
  assert.equal(p.nodes.comparison.hidden,false);
  assert.equal(p.nodes.train.disabled,false);
  assert(p.requests.filter(r=>r.url.endsWith('/job')).every(r=>r.body.job_id==='r'));

  p=await page({statuses:Array(5).fill('network')});
  for(let i=0;i<4;i++){p.timers.shift()();await flush();}
  assert.equal(p.timers.length,0);
  assert.equal(p.nodes['check-training'].hidden,false);
  assert.equal(p.nodes.train.disabled,true);
  p.nodes['check-training'].listeners.click(); await flush();
  assert.equal(p.nodes.comparison.hidden,false);
  assert(!p.requests.some(r=>r.url.endsWith('/start')));
  p=await page();
  assert.equal(p.history.address,'/train');
  assert.equal(p.history.state.poisonGuardTraining.version,'v');
  const remembered=JSON.parse(p.storage.get('poison-guard-training'));
  p=await page({search:'',saved:remembered});
  assert.equal(p.requests[0].url,'/api/training/prepared');
  assert.equal(p.requests[0].body.version,'v');
  assert(p.requests.some(r=>r.url.endsWith('/job') && r.body.job_id==='r'));
  p=await page({search:'',entry:{poisonGuardTraining:{version:'entry-v',scan:'s'}},saved:{version:'stale'}});
  assert.equal(p.requests[0].body.version,'entry-v');
  p=await page({saved:{version:'stale',run:'stale-run'}});
  assert.equal(p.requests[0].body.version,'v');
  p=await page({search:''});
  assert.equal(p.requests.length,0);
  assert(p.nodes['preparation-status'].textContent.includes('completed scan'));

  p=await page({statuses:[{version:'v',status:'error',message:'Training was interrupted by a server restart.'}]});
  assert.equal(p.nodes.train.disabled,false);
  assert(p.nodes['training-status'].textContent.includes('interrupted'));
  assert.equal(p.timers.length,0);
  console.log('Passed: precision/recall, undefined rates, connection recovery and interrupted runs.');
})().catch(error=>{console.error(error);process.exitCode=1});
