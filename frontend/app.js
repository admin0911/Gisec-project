const $ = id => document.getElementById(id);
// Leila: edited settings invalidate progress and results from an earlier build.
let buildRevision = 0;
// Leila: keep the build settings fixed while the extraction request is running.
let buildingDataset = false;
// Leila: set false later to restore MNIST encoder extraction; the shared backend is preserved.
const MNIST_PIXELS_ONLY = true;
function updateAttackOptions() {
  // Leila: keep the shared encoder request, but hide its selector for MNIST pixel scanning.
  $('encoder-control').hidden = true;
  // Leila: enable the verified phrase-backdoor build for IMDB.
  const textDataset = $('dataset').value === 'imdb';
  const cifarDataset = $('dataset').value === 'cifar10';
  for (const option of $('attack').options) {
    option.hidden = textDataset && !['none','label_flip','backdoor'].includes(option.value);
    option.disabled = option.hidden;
    if (option.value === 'backdoor') option.textContent = textDataset ? 'Backdoor phrase' : 'Backdoor patch';
  }
  if (textDataset && !['none','label_flip','backdoor'].includes($('attack').value)) $('attack').value='none';
  $('attack').disabled = false;
  $('poison-rate').disabled = $('attack').value === 'none';
  const targeted = $('attack').value === 'targeted_label_flip';
  // Leila: mixed attacks expose the shared trigger controls.
  const mixed = ['mixed_noise','mixed_all'].includes($('attack').value);
  if ($('mixed-note')) $('mixed-note').hidden = !mixed;
  const blended = $('attack').value === 'blended_injection' || mixed;
  const targetRequired = targeted || blended || $('attack').value === 'backdoor';
  $('source-control').hidden = !targeted;
  $('target-control').hidden = !targetRequired;
  $('alpha-control').hidden = !blended;
  // Leila: offer 7% for both label-flip and patch-backdoor builds.
  const labelFlip = ['label_flip', 'targeted_label_flip', 'backdoor', 'mixed_noise', 'mixed_all'].includes($('attack').value);
  $('rate-seven').hidden = !labelFlip; $('rate-seven').disabled = !labelFlip;
  if (!labelFlip && $('poison-rate').value === '0.07') $('poison-rate').value = '0.05';
  updateWorkflowVisibility();
}
$('dataset').onchange = updateAttackOptions;
$('attack').onchange = updateAttackOptions;
function updateWorkflowVisibility() {
  const attack = $('attack').value;
  const cifar = $('dataset').value === 'cifar10';
  const labelFlip = ['label_flip', 'targeted_label_flip'].includes(attack);
  const blended = attack === 'blended_injection' && cifar;
  $('label-flip-controls').hidden = !labelFlip;
  if ($('backdoor-controls')) $('backdoor-controls').hidden = attack !== 'backdoor';
  $('blended-controls').hidden = !blended;
  if (!blended) $('blended-result').hidden = true;
  if (!labelFlip) {
    $('last-scan').hidden = true;
  }
}
$('scope').onchange = () => {
  const fullTraining = $('scope').value === 'full';
  $('sample-control').hidden = fullTraining;
};
updateAttackOptions();
function clearBuildStatus() {
  buildRevision++;
  $('status').textContent = '';
  $('progress').value = 0; $('progress').hidden = true;
  $('progress-copy').textContent = ''; $('progress-copy').hidden = true;
  $('result').hidden = true;
}
document.querySelectorAll('.controls input, .controls select').forEach(control => {
  control.addEventListener('input',clearBuildStatus);
  control.addEventListener('change',clearBuildStatus);
});
function draw(points, labels) {
  const canvas = $('plot'), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  points.forEach((point, index) => {
    const x = 20 + ((point[0] - minX) / (maxX - minX || 1)) * (canvas.width - 40);
    const y = 20 + ((point[1] - minY) / (maxY - minY || 1)) * (canvas.height - 40);
    ctx.fillStyle = `hsl(${(labels[index] * 37) % 360} 70% 65%)`;
    ctx.beginPath(); ctx.arc(x, canvas.height - y, 4, 0, Math.PI * 2); ctx.fill();
  });
}
$('extract').onclick = async () => {
  // Leila: prevent duplicate builds and restore each control's original enabled state.
  if (buildingDataset) return;
  buildingDataset = true;
  const buildControls = [...document.querySelectorAll('.controls input, .controls select')];
  const previousDisabled = buildControls.map(control => control.disabled);
  buildControls.forEach(control => { control.disabled = true; });
  const revision = ++buildRevision;
  $('status').textContent = 'Building dataset…';
  $('result').hidden = true; $('progress').value = 0; $('progress-copy').textContent = '';
  $('progress').hidden = false; $('progress-copy').hidden = false;
  $('extract').disabled = true;
  // Leila: only enable scanning if the form still describes this completed extraction.
  const scanSettings = () => JSON.stringify([...document.querySelectorAll('.controls input, .controls select')].map(control => control.value));
  const extractionSettings = scanSettings();
  try {
    const response = await fetch('/api/extract', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset: $('dataset').value,
        pixels_only: MNIST_PIXELS_ONLY && $('dataset').value === 'mnist',
        encoder: $('encoder').value,
        full_training: $('scope').value === 'full',
        limit: Number($('limit').value),
        attack: $('attack').value,
        poison_rate: Number($('poison-rate').value),
        source_label: Number($('source-label').value),
        target_label: Number($('target-label').value),
        blend_alpha: Number($('blend-alpha').value)
      })
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Extraction failed');
    const data = await waitForJob(job.job_id, revision);
    if (revision !== buildRevision) return;
    const representations = data.representations || [data];
    $('summary').textContent = data.pixels_only ? `${data.samples} MNIST images ready · pixel scanning` : representations
      .map(item => `${item.encoder || 'minilm'}: ${item.samples} samples · ${item.feature_dim}D raw · ${item.reduced_dim}D PCA`)
      .join(' | ');
    if (data.attack_counts && ['mixed_noise','mixed_all'].includes($('attack').value)) {
      const names={label_flip:'label flips',backdoor:'patches',blended_injection:'noise'};
      $('summary').textContent += ' · ' + Object.entries(data.attack_counts).map(([k,n])=>`${n} ${names[k] || k}`).join(' · ');
    }
    $('artifacts').textContent = data.feature_file
      ? `Saved for reuse: ${representations.map(item => `${item.feature_file}${item.image_file ? ` · images: ${item.image_file}` : ''}`).join(' | ')}`
      : '';
    // Leila: scan readiness must not depend on the optional visualization.
    if (scanSettings() === extractionSettings) {
      document.dispatchEvent(new CustomEvent('features-ready', {detail: data}));
    }
    const unifiedPipelineRun = data.attack === 'blended_injection' && data.detectors?.blended_injection;
    $('last-scan').href = unifiedPipelineRun
      ? `/scan?job=${encodeURIComponent(job.job_id)}`
      : '/scan';
    $('last-scan').textContent = unifiedPipelineRun ? 'View unified results' : 'Run label-flip scan to view results';
    // Leila: build-only detector results are not a complete dataset scan.
    $('last-scan').hidden = true;
    // Leila: a pixels-only build has no encoder/PCA plot to display.
    $('result').hidden = false;
    $('plot').hidden = !!data.pixels_only;
    if (!data.pixels_only) draw(data.visual_features, data.labels);
    $('status').textContent = data.poisoned === null ? 'Dataset ready' : `Dataset ready · ${data.poisoned} poisoned samples`;
  } catch (error) { if (revision === buildRevision) $('status').textContent = `Error: ${error.message}`; }
  finally {
    // Leila: unlock on success or failure; clean datasets still have poison rate disabled.
    buildControls.forEach((control,index) => { control.disabled = previousDisabled[index]; });
    buildingDataset = false;
    $('extract').disabled = false;
  }
};

async function waitForJob(jobId, revision) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok || job.status === 'error') throw new Error(job.message || 'Extraction failed');
    if (revision === buildRevision) {
      $('progress').value = job.progress;
      $('progress-copy').textContent = `${job.progress}% · ${job.message}`;
    }
    if (job.status === 'complete') return job.result;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}
