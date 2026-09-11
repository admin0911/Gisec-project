(() => {
  const input = document.getElementById('backdoor-scan-features');
  const button = document.getElementById('scan-backdoor');
  const status = document.getElementById('backdoor-status');
  let featureFile = '';
  let running = false;

  function reset() {
    featureFile = '';
    button.disabled = true;
    input.replaceChildren(new Option('Build a dataset first', ''));
    status.textContent = 'Build a dataset first.';
  }

  document.addEventListener('features-ready', event => {
    const data = event.detail;
    const representations = data.representations || [data];
    const feature = representations.find(item => item.feature_file);
    if (!feature) { reset(); return; }
    featureFile = feature.feature_file;
    input.replaceChildren(new Option(`${feature.encoder || 'features'} · ${feature.samples} samples`, featureFile));
    input.value = featureFile;
    button.disabled = false;
    status.textContent = 'Feature bundle ready for the backdoor detector track.';
  });
  document.getElementById('dataset').addEventListener('change', reset);
  button.addEventListener('click', async () => {
    if (!featureFile || running) return;
    running = true; button.disabled = true; status.textContent = 'Running feature backdoor detectors…';
    try {
      // Use the unified scan route so feature findings and Leila's
      // fail-safe detectors share review and training preparation.
      const response = await fetch('/api/label-flip/scan', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({feature_file: featureFile})
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || 'Could not start backdoor scan.');
      window.location.assign(`/scan?job=${encodeURIComponent(job.job_id)}`);
    } catch (error) {
      status.textContent = `Error: ${error.message}`;
      button.disabled = false;
    } finally { running = false; }
  });
})();
