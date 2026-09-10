const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(path.join(__dirname,'..','frontend','app.js'),'utf8');
function page(hold=false,fail=false) {
  const ids=['dataset','encoder','scope','limit','attack','poison-rate','target-label','blend-alpha'];
  const nodes={},events=[],requests=[]; let release;
  const element=()=>({value:'',textContent:'Ready',hidden:false,disabled:false,listeners:{},addEventListener(n,f){this.listeners[n]=f}});
  for(const id of [...ids,'encoder-control','target-control','alpha-control','sample-control','rate-seven','status','progress','progress-copy','result','summary','artifacts','extract','plot'])nodes[id]=element();
  for(const [id,value] of Object.entries({dataset:'cifar10',encoder:'resnet18',scope:'full',limit:'100',attack:'none','poison-rate':'0.05','target-label':'0','blend-alpha':'0.10'})) nodes[id].value=value;
  nodes.attack.options=['none','label_flip','backdoor','blended_injection'].map(value=>({value}));
  nodes.plot.getContext=()=>({clearRect(){},beginPath(){},arc(){},fill(){}});
  const payload={dataset:'cifar10',samples:1,feature_dim:512,reduced_dim:2,poisoned:0,visual_features:[[0,0]],labels:[0],feature_file:'saved'};
  vm.runInNewContext(code,{document:{getElementById:id=>nodes[id],querySelectorAll:()=>ids.map(id=>nodes[id]),dispatchEvent:e=>events.push(e)},
    CustomEvent:class{constructor(name,options){this.name=name;this.detail=options.detail}},setTimeout,
    fetch:async(url,options)=>{requests.push({url,body:options?JSON.parse(options.body):null});if(hold&&options)await new Promise(resolve=>release=resolve);
      if(fail) throw new Error('Build failed');
      return{ok:true,json:async()=>options?{job_id:'j'}:{status:'complete',progress:100,message:'Done',result:payload}};}});
  return{nodes,events,requests,release:()=>release()};
}
(async()=>{
  let p=page();
  // Leila: the shared extraction request remains, but MNIST does not expose encoder choices.
  p.nodes.dataset.value='mnist';p.nodes.dataset.onchange();assert(p.nodes['encoder-control'].hidden);
  assert(p.nodes['rate-seven'].hidden);
  p.nodes.attack.value='label_flip';p.nodes.attack.onchange();
  assert(!p.nodes['rate-seven'].disabled);
  p.nodes['poison-rate'].value='0.07';
  await p.nodes.extract.onclick();
  assert.equal(p.requests[0].body.poison_rate,.07);
  assert(p.nodes.status.textContent.includes('Dataset ready'));
  assert.equal(p.events.length,1);
  for(const id of ['dataset','encoder','scope','limit','attack','poison-rate','target-label','blend-alpha']) {
    p.nodes.status.textContent='Ready';p.nodes.result.hidden=false;p.nodes.progress.hidden=false;
    p.nodes[id].listeners.input();
    assert.equal(p.nodes.status.textContent,'');assert(p.nodes.result.hidden);assert(p.nodes.progress.hidden);
  }
  p.nodes.attack.value='backdoor';p.nodes.attack.onchange();
  assert.equal(p.nodes['poison-rate'].value,'0.07');assert(!p.nodes['rate-seven'].disabled);
  await p.nodes.extract.onclick();
  assert.equal(p.requests.findLast(r=>r.body).body.poison_rate,.07);
  assert.equal(p.requests.findLast(r=>r.body).body.attack,'backdoor');
  p.nodes.attack.value='blended_injection';p.nodes.attack.onchange();
  assert.equal(p.nodes['poison-rate'].value,'0.05');assert(p.nodes['rate-seven'].disabled);
  p=page(true);const pending=p.nodes.extract.onclick();
  // Leila: every build setting is locked, and duplicate clicks cannot start another job.
  for(const id of ['dataset','encoder','scope','limit','attack','poison-rate','target-label','blend-alpha']) assert(p.nodes[id].disabled);
  await p.nodes.extract.onclick();assert.equal(p.requests.length,1);
  p.nodes.limit.value='200';p.nodes.limit.listeners.input();
  p.release();await pending;
  assert.equal(p.nodes.status.textContent,'');assert(p.nodes.result.hidden);
  assert.equal(p.events.length,0);assert.equal(p.nodes.extract.disabled,false);
  assert.equal(p.nodes.dataset.disabled,false);assert.equal(p.nodes['poison-rate'].disabled,true);
  p=page(true,true);p.nodes.attack.value='label_flip';p.nodes.attack.onchange();
  const failed=p.nodes.extract.onclick();assert(p.nodes['poison-rate'].disabled);
  p.release();await failed;
  for(const id of ['dataset','encoder','scope','limit','attack','poison-rate','target-label','blend-alpha']) assert.equal(p.nodes[id].disabled,false);
  assert(p.nodes.status.textContent.includes('Error: Build failed'));
  assert.equal(p.nodes.extract.disabled,false);
  console.log('Passed: 7% label flip and backdoor, reset on every setting, and stale build completion.');
})().catch(e=>{console.error(e);process.exitCode=1});
