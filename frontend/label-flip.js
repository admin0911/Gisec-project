// Leila: navigation uses the short scan route with the selected job preserved.
// Label-flip controls are separate from the existing extraction controller.
(() => {
  const button = document.getElementById('scan-label-flips');
  const message = document.getElementById('scan-readiness');
  const previous = document.getElementById('last-scan');
  const saved = document.getElementById('saved-scan-features');
  // Leila: readable descriptions only; option values retain the exact saved paths.
  function datasetLabel(file) {
    const name = String(file).split(/[\\/]/).pop();
    const header = name.match(/^(cifar10|mnist|imdb)-(train|test)-([^-]+)-/);
    if (!header) return 'Saved dataset · details unavailable';
    const dataset = {cifar10:'CIFAR-10',mnist:'MNIST',imdb:'IMDB'}[header[1]];
    const size = header[3] === 'full' ? `Full ${header[2] === 'train' ? 'training' : 'test'} dataset` : `${Number(header[3]).toLocaleString()} ${header[1] === 'imdb' ? 'reviews' : 'images'}`;
    const attack = name.match(/-(targeted_label_flip|label_flip|blended_injection|mixed_noise|mixed_all|backdoor|none)-/);
    const kind = attack?.[1];
    const attacks = {mixed_noise:'Mixed: label flip + noise',mixed_all:'Mixed: label flip + patch + noise',none:'Clean',label_flip:'Random label flip',targeted_label_flip:'Targeted label flip',backdoor:header[1] === 'imdb'?'Backdoor phrase':'Backdoor patch',blended_injection:'Blended noise'};
    const parts = [dataset, attacks[kind] || 'Attack unspecified'];
    const count = name.match(/-n(\d+)-/);
    const rate = name.match(/-(\d{3})-seed/);
    if (kind && kind !== 'none') {
      if (count) parts.push(`${Number(count[1]).toLocaleString()} poisoned samples`);
      else if (rate) parts.push(`${Number(rate[1])}% poisoning`);
    }
    parts.push(size);
    const source = name.match(/-s(\d+)-/), target = name.match(/-t(\d+)-/);
    if (kind === 'targeted_label_flip' && source && target) parts.push(`Label ${source[1]} → ${target[1]}`);
    else if (['backdoor','blended_injection','mixed_noise','mixed_all'].includes(kind) && target) parts.push(`Target ${target[1]}`);
    const alpha = name.match(/-a([\d.]+)-/);
    if (['blended_injection','mixed_noise','mixed_all'].includes(kind) && alpha) parts.push(`Blend ${Number(alpha[1])}`);
    const seed = name.match(/-seed(\d+)/);
    if (seed) parts.push(`Seed ${seed[1]}`);
    return parts.join(' · ');
  }
  let featureFile = null, generation = 0;
  const last = sessionStorage.getItem('label-flip-job');
  if (last) {
    previous.href = `/scan?job=${encodeURIComponent(last)}`;
    previous.textContent = 'View last label-flip scan';
    previous.hidden = false;
  }
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
      saved.replaceChildren(new Option('Choose saved dataset…', ''));
      for (const pair of data.pairs) saved.add(new Option(datasetLabel(pair.feature_file), pair.feature_file));
      const match = data.pairs.find(pair => pair.feature_file === preferred);
      const selected = match?.feature_file || preferred || (autoSelect ? data.pairs[0]?.feature_file : null);
      if (selected && ![...saved.options].some(option => option.value === selected)) {
        saved.add(new Option(datasetLabel(selected), selected));
      }
      saved.value = selected || '';
      if (selected) await check(selected);
      else { button.disabled = true; message.textContent = 'Choose a saved extraction with at least 21 balanced samples.'; }
    } catch (error) { if (current === generation) message.textContent = error.message; }
  }
  saved.addEventListener('change', () => check(saved.value));
  // Leila: a completed build can be checked directly; listing saved files is independent.
  document.addEventListener('features-ready', event => {
    const feature = event.detail.feature_file;
    if (!feature) { reset(); return; }
    if (![...saved.options].some(option => option.value === feature)) {
      saved.add(new Option(datasetLabel(feature),feature));
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
