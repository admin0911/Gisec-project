// Label-flip controls are separate from the existing extraction controller.
(() => {
  const button = document.getElementById('scan-label-flips');
  const message = document.getElementById('scan-readiness');
  const previous = document.getElementById('last-scan');
  const saved = document.getElementById('saved-scan-features');
  let featureFile = null, generation = 0;
  const last = sessionStorage.getItem('label-flip-job');
  if (last) { previous.href = `/scan-results.html?job=${encodeURIComponent(last)}`; previous.hidden = false; }
  function reset() {
    generation++; featureFile = null; button.disabled = true;
    saved.value = '';
    message.textContent = 'Choose a saved feature pair below, or extract features for the new settings.';
  }
  // Encoder changes allow extracting the other representation; a fresh completion is still required.
  document.querySelectorAll('.controls input, .controls select').forEach(control => control.addEventListener('change', reset));
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
  async function loadSaved(preferred = null) {
    const current = ++generation;
    try {
      const response = await fetch('/api/label-flip/saved', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
      if (!response.ok) throw new Error('Could not load saved features. Restart the updated server.');
      const data = await response.json();
      if (current !== generation) return;
      saved.replaceChildren(new Option('Choose saved features…', ''));
      for (const pair of data.pairs) saved.add(new Option(pair.label, pair.feature_file));
      const match = data.pairs.find(pair => pair.feature_file === preferred);
      const selected = match?.feature_file || preferred || data.pairs[0]?.feature_file;
      if (selected && ![...saved.options].some(option => option.value === selected)) {
        saved.add(new Option('Just extracted features', selected));
      }
      saved.value = selected || '';
      if (selected) await check(selected);
      else { button.disabled = true; message.textContent = 'No complete saved pair yet. Extract both encoders with identical settings.'; }
    } catch (error) { if (current === generation) message.textContent = error.message; }
  }
  saved.addEventListener('change', () => check(saved.value));
  document.addEventListener('features-ready', event => loadSaved(event.detail.feature_file));
  loadSaved();
  button.addEventListener('click', async () => {
    if (!featureFile) return;
    button.disabled = true;
    message.textContent = 'Starting label-flip scan…';
    try {
      const response = await fetch('/api/label-flip/scan', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({feature_file:featureFile})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not start scan.');
      sessionStorage.setItem('label-flip-job', data.job_id);
      window.location.assign(`/scan-results.html?job=${encodeURIComponent(data.job_id)}`);
    } catch (error) { message.textContent = error.message; button.disabled = !featureFile; }
  });
})();
