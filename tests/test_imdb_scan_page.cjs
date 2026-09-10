// Leila: IMDB renders MiniLM votes without image review or training controls.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(path.join(__dirname,'../frontend/scan-results.js'),'utf8');
const nodes={};
const element=()=>({children:[],style:{},hidden:false,appendChild(x){this.children.push(x)},removeAttribute(k){delete this[k]}});
const result={dataset:'imdb',samples:30,summary:{not_flagged:27,uncertain:1,suspected_label_flip:2},
  detectors:[{encoder:'minilm',detector:'knn',flagged:2,rate:2/30,threshold:null,threshold_label:'≥ 0.95 (19/20)'}],
  examples:[{sample_id:'mnist-train:0',label:0,assessment:'uncertain',text_votes:1}],profile:'pixels',limitation:'Provisional'};
vm.runInNewContext(code,{document:{getElementById:id=>nodes[id]||(nodes[id]=element()),createElement:element},
  requestAnimationFrame:fn=>fn(),location:{search:'?job=test'},history:{replaceState(){}},sessionStorage:{getItem(){},setItem(){}},URLSearchParams,
  window:{addEventListener(){}},fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/jobs/') ?
    {status:'complete',message:'Done',progress:100,result} : {saved:0,unreviewed:3,keep:0,quarantine:0,unsure:0}})});
setImmediate(()=>{
  assert(nodes['scan-summary'].textContent.includes('IMDB'));
  assert(nodes['scan-description'].textContent.includes('label flipping in review text'));
  assert.equal(nodes['prepare-training'].hidden,false);
  assert.equal(nodes['human-review'].hidden,false);

  assert(nodes['scan-examples'].children[0].textContent.includes('MiniLM 1/3'));
  assert(!nodes['scan-examples'].children[0].textContent.includes('undefined'));
  assert.equal(nodes['detector-rows'].children[0].children[4].textContent,'≥ 0.95 (19/20)');
  console.log('Passed: MNIST result captions, threshold rules, examples and training availability.');
});
