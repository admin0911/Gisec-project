// Leila: render an MNIST scan without presenting pixel votes as two encoder results.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(path.join(__dirname,'../frontend/scan-results.js'),'utf8');
const nodes={};
const element=()=>({children:[],hidden:false,appendChild(x){this.children.push(x)}});
const result={dataset:'mnist',samples:30,summary:{not_flagged:27,uncertain:1,suspected_label_flip:2},
  detectors:[{encoder:'pixels',detector:'knn',flagged:2,rate:2/30,threshold:null,threshold_label:'≥ 0.95 (19/20)'}],
  examples:[{sample_id:'mnist-train:0',label:0,assessment:'uncertain',pixel_votes:1}],profile:'pixels',limitation:'Provisional'};
result.patch_scan={flagged:4,rate:4/30,overlap:1,unique_flagged:6,patterns:[],examples:[{sample_id:'mnist-train:4',label:7,score:.8}],limitation:'Provisional patch findings'};
vm.runInNewContext(code,{document:{getElementById:id=>nodes[id]||(nodes[id]=element()),createElement:element},
  location:{search:'?job=test'},history:{replaceState(){}},sessionStorage:{getItem(){},setItem(){}},URLSearchParams,
  window:{addEventListener(){}},fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/jobs/') ?
    {status:'complete',message:'Done',progress:100,result} : {saved:0,unreviewed:3,keep:0,quarantine:0,unsure:0}})});
setImmediate(()=>{
  assert(nodes['scan-summary'].textContent.includes('MNIST'));
  assert(nodes['scan-description'].textContent.includes('pixels'));
  assert.equal(nodes['prepare-training'].hidden,false);
  assert.equal(nodes['prepare-training'].target,'_blank');
  assert(nodes['prepare-training'].href.includes('dataset=mnist'));
  assert(nodes['scan-examples'].children[0].textContent.includes('Pixels 1/3'));
  assert(!nodes['scan-examples'].children[0].textContent.includes('undefined'));
  assert.equal(nodes['detector-rows'].children[0].children[4].textContent,'≥ 0.95 (19/20)');
  assert.equal(nodes['patch-results'].hidden,false);
  assert.equal(nodes['patch-not-flagged'].textContent,'26');
  assert.equal(nodes['patch-needs-review'].textContent,'4');
  assert.equal(nodes['patch-details'].hidden,false);
  assert(nodes['patch-examples'].children[0].textContent.includes('label 7'));
  assert.equal(nodes['suspected'].textContent,'2');
  console.log('Passed: MNIST result captions, threshold rules, examples and training availability.');
});
