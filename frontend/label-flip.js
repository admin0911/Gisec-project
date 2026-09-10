// Leila: navigation uses the short scan route with the selected job preserved.
// Label-flip controls are separate from the existing extraction controller.
(() => {
  const button = document.getElementById('scan-label-flips');
  const message = document.getElementById('scan-readiness');
  const previous = document.getElementById('last-scan');
  const saved = document.getElementById('saved-scan-features');
  let featureFile = null, generation = 0;
  const last = sessionStorage.getItem('label-flip-job');
  if (last) { previous.href = `/scan?job=${encodeURIComponent(last)}`; previous.hidden = false; }
  function reset() {
    generation++; featureFile = null; button.disabled = true;
    saved.value = '';
    message.textContent = 'Choose saved data below, or build the dataset for the new settings.';
  }
  // Encoder changes allow extracting the other representation; a fresh completion is still required.
  // Leila: typed edits also invalidate readiness before the field loses focus.
  document.querySelectorAll('.controls input, .controls select').forEach(control => {
    control.addEventListener('change', reset);
    control.addEventListener('input', reset);
  });
  document.getElementById('extract').addEventListener('click', reset);
  async function check(feature) {
    const current = ++generation;
    featureFile = null;
    button.disabled = true;
    if (!feature) { message.textContent = 'Choose saved features for scanning.'; return; }
    message.textContent = 'Checking saved feature files…';
    try {
      const response = await fetch('/api/label-flip/ready', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({feature_file:feature})});
      const data = await response.json();
      if (current !== generation) return;
      if (!response.ok) throw new Error(data.error || 'Could not check feature files.');
      message.textContent = data.message;
      featureFile = data.ready ? feature : null;
      button.disabled = !data.ready;
    } catch (error) { if (current === generation) message.textContent = error.message; }
  }
  async function loadSaved(preferred = null, autoSelect = true) {
    const current = ++generation;
    try {
      const response = await fetch('/api/label-flip/saved', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
      if (!response.ok) throw new Error('Could not load saved features. Restart the updated server.');
      const data = await response.json();
      if (current !== generation) return;
      // Leila: show saved inputs for the selected dataset, including MNIST image connectors.
      const dataset = document.getElementById('dataset').value;
      data.pairs = data.pairs.filter(pair => pair.label.startsWith(dataset + '-'));
      saved.replaceChildren(new Option('Choose saved features…', ''));
      for (const pair of data.pairs) saved.add(new Option(pair.label, pair.feature_file));
      const match = data.pairs.find(pair => pair.feature_file === preferred);
      const selected = match?.feature_file || preferred || (autoSelect ? data.pairs[0]?.feature_file : null);
      if (selected && ![...saved.options].some(option => option.value === selected)) {
        saved.add(new Option('Just extracted features', selected));
      }
      saved.value = selected || '';
      if (selected) await check(selected);
      else { button.disabled = true; message.textContent = 'Choose saved data above or build the dataset first.'; }
    } catch (error) { if (current === generation) message.textContent = error.message; }
  }
  saved.addEventListener('change', () => check(saved.value));
  // Leila: a completed build can be checked directly; listing saved files is independent.
  document.addEventListener('features-ready', event => {
    const feature = event.detail.feature_file;
    if (!feature) { reset(); return; }
    if (![...saved.options].some(option => option.value === feature)) {
      saved.add(new Option('Just built dataset',feature));
    }
    saved.value = feature;
    check(feature);
  });
  // Leila: changing datasets refreshes the list without choosing stale scan inputs.
  document.getElementById('dataset').addEventListener('change', () => loadSaved(null,false));
  loadSaved();
  button.addEventListener('click', async () => {
    if (!featureFile) return;
    button.disabled = true;
    // Leila: matching completed scans are reopened before running detectors again.
    message.textContent = 'Checking for a saved scan…';
    try {
      const response = await fetch('/api/label-flip/scan', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({feature_file:featureFile})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not start scan.');
      sessionStorage.setItem('label-flip-job', data.job_id);
      window.location.assign(`/scan?job=${encodeURIComponent(data.job_id)}`);
    } catch (error) { message.textContent = error.message; button.disabled = !featureFile; }
  });
})();
