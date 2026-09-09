(() => {
  const $ = id => document.getElementById(id), params = new URLSearchParams(location.search);
  const job = params.get('job');
  let page = 0, pages = 1, revision = 0, busy = false, loaded = false;
  let group = params.get('group') === 'suspected_label_flip' ? 'suspected_label_flip' : 'uncertain';
  let size = 20, changes = {};
  $('group').value = group;
  $('back').href = `/scan-results.html?job=${encodeURIComponent(job || '')}`;
  function controls(value) {
    busy = value;
    // Leila: prevent duplicate Finish actions while a save or page request is pending.
    for (const id of ['group','size','previous','save','next','finish']) $(id).disabled = value;
    document.querySelectorAll('#review-grid input').forEach(input => input.disabled = value);
    if (!value) { $('previous').disabled = !loaded || page === 0; $('save').disabled = !loaded; $('next').disabled = !loaded || page + 1 >= pages; }
    $('review-grid').setAttribute('aria-busy', String(value));
  }
  async function api(route, body) {
    const response = await fetch(`/api/review/${route}`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:job,...body})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Review request failed.');
    return result;
  }
  function render(data) {
    // Leila: pending choices survive paging; retain their original revision for conflict checks.
    if (!Object.keys(changes).length) revision = data.revision;
    pages = data.pages; loaded = true;
    $('counts').textContent = `${data.total.toLocaleString()} samples · ${data.resolved} resolved · ${data.unsure} unsure · ${data.unreviewed} unreviewed`;
    $('page-info').textContent = `Page ${page+1} of ${pages}`;
    $('review-grid').replaceChildren();
    data.items.forEach((item,index) => {
      const tile = document.createElement('article'); tile.className = 'review-tile';
      const image = document.createElement('img'); image.src = item.image; image.alt = `Sample ${item.sample_id}, supplied label ${item.class_name}`; image.loading = 'lazy';
      const title = document.createElement('h2'); title.textContent = `${item.class_name} · label ${item.label}`;
      const id = document.createElement('p'); id.textContent = item.sample_id;
      const votes = document.createElement('p'); votes.textContent = `ResNet18 ${item.resnet_votes}/3 · DINOv2 ${item.dino_votes}/3`;
      const field = document.createElement('fieldset'), legend = document.createElement('legend'); legend.textContent = 'Your decision'; field.appendChild(legend);
      for (const [value,text] of [['keep','Keep'],['quarantine','Quarantine'],['unsure','Unsure']]) {
        const label = document.createElement('label'), input = document.createElement('input');
        input.type = 'radio'; input.name = `decision-${index}`; input.value = value;
        input.checked = (changes[item.sample_id] ?? item.decision) === value;
        input.addEventListener('change', () => { changes[item.sample_id] = value; $('review-status').textContent = `${Object.keys(changes).length} unsaved choice(s).`; });
        label.append(input,document.createTextNode(text)); field.appendChild(label);
      }
      tile.append(image,title,id,votes,field); $('review-grid').appendChild(tile);
    });
    if (!data.items.length) $('review-grid').textContent = 'No samples in this group.';
  }
  async function load() {
    controls(true); $('review-status').textContent = 'Loading review images…';
    try {
      render(await api('page',{group,page,page_size:size}));
      const pending = Object.keys(changes).length;
      $('review-status').textContent = pending ? `${pending} unsaved choice(s) across pages. Use Save review when ready.` : 'Choose a decision for any images you want to review.';
    }
    catch(error) { loaded = false; $('review-grid').replaceChildren(); $('review-status').textContent = error.message; }
    finally { controls(false); }
  }
  async function save() {
    // Leila: an explicit save includes all visited pages, within the API's 100-choice limit.
    while (Object.keys(changes).length) {
      const batch = Object.fromEntries(Object.entries(changes).slice(0,100));
      const result = await api('save',{changes:batch,revision});
      revision = result.revision;
      for (const id of Object.keys(batch)) delete changes[id];
    }
  }
  async function navigate(nextPage, nextGroup=group, nextSize=size) {
    if (busy) return;
    controls(true);
    // Leila: navigation only changes the view; it never saves review choices.
    try { page = nextPage; group = nextGroup; size = nextSize; await load(); }
    catch(error) { $('review-status').textContent = error.message; $('group').value = group; $('size').value = size; }
    finally { controls(false); }
  }
  $('group').addEventListener('change', () => navigate(0,$('group').value,size));
  $('size').addEventListener('change', () => navigate(0,group,Number($('size').value)));
  $('previous').addEventListener('click', () => navigate(page-1));
  $('next').addEventListener('click', () => navigate(page+1));
  $('save').addEventListener('click', async () => {
    if (busy) return;
    controls(true);
    try { await save(); await load(); if (loaded) $('review-status').textContent = 'Review saved across all visited pages. Unselected samples remain unreviewed.'; }
    catch (error) { $('review-status').textContent = `${error.message} Unsaved choices are still pending.`; }
    finally { controls(false); }
  });
  // Leila: save pending review choices before returning, avoiding the unsaved-changes prompt.
  async function finishReview() {
    if (busy) { $('review-status').textContent = 'Please wait for the current request to finish, then return to Scan results.'; return; }
    controls(true);
    $('review-status').textContent = 'Saving choices and opening Scan results…';
    try {
      await save();
      window.location.assign($('back').href);
    } catch (error) {
      $('review-status').textContent = `Could not return: ${error.message} Your choices are still on this page.`;
      controls(false);
    }
  }
  // Leila: confirm pending changes on Finish; Cancel leaves choices unsaved and intact.
  async function requestFinish() {
    if (busy) return;
    if (Object.keys(changes).length) $('finish-confirm').showModal();
    else await finishReview();
  }
  $('finish').addEventListener('click', requestFinish);
  $('finish-cancel').addEventListener('click', () => $('finish-confirm').close());
  // Leila: Don't save clears local pending edits and returns without writing to the server.
  $('finish-discard').addEventListener('click', () => {
    if (busy) return;
    $('finish-confirm').close();
    changes = {};
    controls(true);
    window.location.assign($('back').href);
  });
  $('finish-save').addEventListener('click', async () => {
    $('finish-confirm').close();
    await finishReview();
  });
  $('back').addEventListener('click', async event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button > 0) return;
    event.preventDefault();
    await requestFinish();
  });
  window.addEventListener('beforeunload', event => { if (Object.keys(changes).length) { event.preventDefault(); event.returnValue = ''; } });
  load();
})();
