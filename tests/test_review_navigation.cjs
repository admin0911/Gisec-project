// Exercise the real review controller without changing any user reviews.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const code = fs.readFileSync(path.join(__dirname, '../frontend/human-review.js'), 'utf8');
async function scenario(change, failure, action='back', cancel=false, browse=false) {
  const created = [], nodes = {}, events = {}, calls = [], locations = [];
  function element(tag) {
    const e = {tag, value:'', handlers:{}, children:[],
      addEventListener(name, fn) { this.handlers[name] = fn; },
      append(...items) { this.children.push(...items); },
      appendChild(item) { this.children.push(item); },
      replaceChildren(...items) { this.children = items; }, setAttribute() {},
      showModal() { this.open = true; }, close() { this.open = false; }};
    created.push(e); return e;
  }
  const context = {
    URLSearchParams, location:{search:'?job='+'a'.repeat(32)},
    document:{getElementById:id => nodes[id] ||= element(id), createElement:element,
      createTextNode:text => text, querySelectorAll:() => []},
    window:{location:{assign:url => locations.push(url)}, addEventListener:(name,fn) => events[name] = fn},
    fetch:async (route, options) => {
      calls.push({route,body:JSON.parse(options.body)});
      if (route.endsWith('/save')) return {ok:!failure,json:async () => failure ? {error:'Save failed'} : {revision:1}};
      const page = JSON.parse(options.body).page;
      return {ok:true,json:async () => ({revision:0,pages:2,total:2,resolved:0,unsure:0,unreviewed:2,
        items:[{sample_id:`sample:${page+1}`,label:0,class_name:'airplane',image:'data:image/png;base64,test',resnet_votes:2,dino_votes:0,decision:null}]})};
    }
  };
  vm.runInNewContext(code,context);
  await new Promise(resolve => setImmediate(resolve));
  if (change) created.find(e => e.tag === 'input' && e.value === 'keep').handlers.change();
  if (browse) {
    await nodes.next.handlers.click();
    assert.equal(calls.filter(c => c.route.endsWith('/save')).length,0);
    assert.equal(calls.at(-1).body.page,1);
    created.findLast(e => e.tag === 'input' && e.value === 'quarantine').handlers.change();
    await nodes.previous.handlers.click();
    assert.equal(calls.filter(c => c.route.endsWith('/save')).length,0);
    assert.equal(created.findLast(e => e.tag === 'input' && e.value === 'keep').checked,true);
    assert.match(nodes['review-status'].textContent,/2 unsaved/);
  }
  let prevented = false;
  await nodes[action].handlers.click({button:0,preventDefault() { prevented = true; }});
  assert.equal(prevented,action === 'back');
  if (change) {
    assert.equal(nodes['finish-confirm'].open,true);
    assert.equal(calls.filter(c => c.route.endsWith('/save')).length,0);
    assert.equal(locations.length,0);
    await nodes[cancel === 'discard' ? 'finish-discard' : cancel ? 'finish-cancel' : 'finish-save'].handlers.click();
    assert.equal(nodes['finish-confirm'].open,false);
    if (cancel) {
      assert.equal(calls.filter(c => c.route.endsWith('/save')).length,0);
      assert.equal(locations.length,cancel === 'discard' ? 1 : 0);
      if (cancel === 'discard') assert.equal(locations[0],'/scan-results.html?job='+'a'.repeat(32));
      let blocked = false;
      events.beforeunload({preventDefault() { blocked = true; }});
      assert.equal(blocked,cancel !== 'discard');
      return;
    }
  }
  const saves = calls.filter(c => c.route.endsWith('/save'));
  assert.equal(saves.length,change ? 1 : 0);
  if (change) assert.equal(saves[0].body.changes['sample:1'],'keep');
  if (browse) assert.equal(saves[0].body.changes['sample:2'],'quarantine');
  if (failure) {
    assert.equal(locations.length,0);
    assert.match(nodes['review-status'].textContent,/Your choices are still on this page/);
  } else {
    assert.deepEqual(locations,['/scan-results.html?job='+'a'.repeat(32)]);
    let blocked = false;
    events.beforeunload({preventDefault() { blocked = true; }});
    assert.equal(blocked,false);
  }
}
(async () => {
  await scenario(false,false);
  await scenario(true,false);
  await scenario(true,true);
  await scenario(false,false,'finish');
  await scenario(true,false,'finish');
  await scenario(true,true,'finish');
  await scenario(true,false,'finish',true);
  await scenario(true,false,'finish',false,true);
  await scenario(true,false,'finish','discard',true);
  console.log('Passed: pending edits survive paging; Save, Cancel and Don’t save behave correctly.');
})().catch(error => { console.error(error); process.exitCode = 1; });
