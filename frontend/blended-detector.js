(() => {
  const input = document.getElementById('blended-scan-images');
  const button = document.getElementById('scan-blended');
  const status = document.getElementById('blended-status');
  const result = document.getElementById('blended-result');
  const chart = document.getElementById('blended-chart');
  let scanning = false;
  let imageFile = '';

  function reset() {
    imageFile = '';
    button.disabled = true;
    result.hidden = true;
    status.textContent = 'Build a CIFAR-10 dataset first.';
    input.replaceChildren(new Option('Build a CIFAR-10 dataset first', ''));
  }

  document.addEventListener('features-ready', event => {
    const data = event.detail;
    const representations = data.representations || [data];
    const image = representations.find(item => item.image_file);
    if (!image || data.dataset !== 'cifar10') {
      reset();
      status.textContent = 'The blended detector currently requires a CIFAR-10 image bundle.';
      return;
    }
    imageFile = image.image_file;
    input.replaceChildren(new Option(`${image.encoder || 'image'} · ${image.samples} images`, imageFile));
    input.value = imageFile;
    button.disabled = false;
    status.textContent = 'Saved blended-injection pixels are ready for analysis.';
    if (document.getElementById('attack').value === 'blended_injection') runScan();
  });

  document.getElementById('dataset').addEventListener('change', reset);
  button.addEventListener('click', async () => {
    runScan();
  });

  async function runScan() {
    if (!imageFile || scanning) return;
    scanning = true;
    button.disabled = true;
    status.textContent = 'Scanning shared residual signatures…';
    try {
      const response = await fetch('/api/blended-injection/scan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image_file: imageFile})
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || 'Could not start blended scan.');
      const data = await waitForJob(job.job_id);
      render(data);
    } catch (error) {
      status.textContent = `Error: ${error.message}`;
    } finally {
      scanning = false;
      button.disabled = false;
    }
  }

  async function waitForJob(jobId) {
    while (true) {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      if (!response.ok || job.status === 'error') throw new Error(job.message || 'Blended scan failed.');
      status.textContent = `${job.progress}% · ${job.message}`;
      if (job.status === 'complete') return job.result;
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  function render(data) {
    const evidence = data.evidence || {};
    const found = evidence.target_class !== null && evidence.target_class !== undefined;
    const flags = data.flags || [];
    const flagged = flags.filter(Boolean).length;
    const contrasts = evidence.class_contrasts || {};
    document.getElementById('blended-target').textContent = found ? evidence.target_class : 'None';
    document.getElementById('blended-flagged').textContent = flagged.toLocaleString();
    document.getElementById('blended-contrast').textContent = Number(evidence.contrast || 0).toFixed(2);
    drawContrastChart(contrasts);
    document.getElementById('blended-summary').textContent = found
      ? `Target class ${evidence.target_class} identified · ${flagged} samples flagged`
      : 'No blended-injection signature identified';
    const ratio = evidence.class_ratio === undefined
      ? '' : ` · class contrast ratio ${Number(evidence.class_ratio).toFixed(2)}`;
    document.getElementById('blended-details').textContent =
      `Contrast ${Number(evidence.contrast || 0).toFixed(3)}${ratio}. ` +
      'Flags are detector findings for review, not proof of poisoning.';
    const examples = document.getElementById('blended-examples');
    examples.replaceChildren();
    flags.forEach((flag, index) => {
      if (!flag) return;
      const p = document.createElement('p');
      p.textContent = `${data.sample_ids[index]} · score ${Number(data.scores[index]).toFixed(3)}`;
      examples.appendChild(p);
    });
    if (!flagged) examples.textContent = 'No samples were flagged.';
    result.hidden = false;
    status.textContent = 'Blended-injection scan complete.';
  }

  function drawContrastChart(contrasts) {
    if (!chart) return;
    const ctx = chart.getContext('2d');
    const entries = Object.entries(contrasts);
    const max = Math.max(1, ...entries.map(([, value]) => Number(value)));
    const width = chart.width, height = chart.height;
    let phase = 0;
    const started = performance.now();
    function frame(now) {
      phase = Math.min(1, (now - started) / 700);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#10131a'; ctx.fillRect(0, 0, width, height);
      const barWidth = entries.length ? (width - 60) / entries.length : width;
      entries.forEach(([label, value], index) => {
        const amount = Math.max(0, Number(value)) / max * (height - 55) * phase;
        const x = 30 + index * barWidth + barWidth * .18;
        const y = height - 28 - amount;
        ctx.fillStyle = Number(label) === Number(evidenceTarget()) ? '#f2c879' : '#72e6c2';
        ctx.fillRect(x, y, barWidth * .64, amount);
        ctx.fillStyle = '#abb6c9'; ctx.font = '12px Segoe UI';
        ctx.fillText(label, x + barWidth * .2, height - 10);
      });
      if (phase < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function evidenceTarget() {
    return document.getElementById('blended-target').textContent;
  }
})();
