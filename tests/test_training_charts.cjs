// Leila: check saved-run chart selection, missing values and PNG export wiring.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const labels=[],nodes={};let downloaded=false;
const ctx=new Proxy({}, {get:(o,k)=>o[k] || (k==='fillText'?(s)=>labels.push(s):()=>{})});
for(const id of ['comparison-chart','chart-summary','result-charts','download-chart','chart-download-status'])nodes[id]={};
nodes['comparison-chart'].getContext=()=>ctx;
nodes['comparison-chart'].toBlob=cb=>cb({});
const sandbox={document:{getElementById:id=>nodes[id],body:{appendChild(){}},createElement:()=>({click(){downloaded=true;},remove(){}})},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},setTimeout:cb=>cb()};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync('frontend/training-charts.js','utf8')+'\nglobalThis.charts=TrainingCharts;',sandbox);
const data={runs:{clean_reference:{metrics:{accuracy:.9}},before_cleaning:{metrics:{accuracy:.8},backdoor_metrics:{asr_non_target:.9}},after_cleaning:{metrics:{accuracy:.85},backdoor_metrics:{asr_non_target:.1}}}};
const macro=()=>({f1:.7});
assert.equal(sandbox.charts.model(data,macro,false).panels[1].title,'Backdoor ASR');
sandbox.charts.render(data,macro,false);
assert(labels.includes('N/A'));assert(labels.includes('90.00%'));
assert(nodes['chart-summary'].textContent.includes('decreased by 80.00'));
nodes['download-chart'].onclick();assert(downloaded);
delete data.runs.before_cleaning.backdoor_metrics;delete data.runs.after_cleaning.backdoor_metrics;
assert.equal(sandbox.charts.model(data,macro,true).panels[1].title,'Macro F1');
assert.equal(sandbox.charts.model(data,macro,true).rows.length,2);
console.log('Passed: chart metrics, missing ASR, skipped arm, change summary and PNG export.');

assert.equal(sandbox.charts.filename({dataset:'mnist'},{attack:'label_flip',poisonPercent:5}), 'poisonguard-mnist-label_flip-5pct-1seed-training.png');

data.benchmark={summary:{after_cleaning:{accuracy:{mean:.91},macro_f1:{mean:.88}}}};
assert.equal(sandbox.charts.model(data,macro,false).panels[0].values[2],.91);
assert.equal(sandbox.charts.model(data,macro,false).panels[1].values[2],.88);
