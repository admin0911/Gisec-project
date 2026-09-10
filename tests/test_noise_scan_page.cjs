// Leila: noise status must remain visible for completed, missing and unsupported stages.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync('frontend/scan-results.js','utf8');
async function page(dataset,blend) {
 const nodes={};
 const element=()=>({children:[],style:{},hidden:false,appendChild(x){this.children.push(x)},replaceChildren(){this.children=[]}});
 const result={dataset,samples:30,summary:{not_flagged:30,uncertain:0,suspected_label_flip:0},detectors:[],examples:[],profile:'test',limitation:'test'};
 if(blend) result.blended_scan=blend;
 vm.runInNewContext(code,{document:{getElementById:id=>nodes[id]||(nodes[id]=element()),createElement:element},
  requestAnimationFrame:fn=>fn(),location:{search:'?job=test'},history:{replaceState(){}},sessionStorage:{getItem(){},setItem(){}},URLSearchParams,
  window:{addEventListener(){}},fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/jobs/')?{status:'complete',message:'Done',progress:100,result}:{saved:0,unreviewed:0,keep:0,quarantine:0,unsure:0}})});
 await new Promise(resolve=>setImmediate(resolve));return nodes;
}
(async()=>{
 for(const dataset of ['cifar10','mnist']) {
  let n=await page(dataset,{applicable:true,status:'completed',flagged:3,method:'residual-signature',settings:{},examples:[]});
  assert.equal(n['blended-results'].hidden,false);assert.equal(n['blended-counts'].hidden,false);
  assert.equal(n['blended-flagged'].textContent,'3');assert.equal(n['blended-status'].textContent,'');assert.equal(n['blended-status'].hidden,true);
  n=await page(dataset);assert.equal(n['blended-results'].hidden,false);assert.equal(n['blended-counts'].hidden,true);
  assert(n['blended-status'].textContent.includes('Not run'));
 }
 const n=await page('imdb');assert.equal(n['blended-results'].hidden,false);
 assert(n['blended-status'].textContent.includes('Not applicable'));assert.equal(n['blended-counts'].hidden,true);
 console.log('Passed: CIFAR/MNIST completed and missing noise stages; IMDB applicability.');
})().catch(e=>{console.error(e);process.exitCode=1});
