const $ = id => document.getElementById(id);
function updateAttackOptions() {
  const imageDataset = $('dataset').value === 'mnist';
  $('encoder-control').hidden = !imageDataset;
  $('attack').disabled = false;
  $('poison-rate').disabled = $('attack').value === 'none';
  const targeted = $('attack').value === 'targeted_label_flip';
  const blended = $('attack').value === 'blended_injection';
  const targetRequired = targeted || blended || $('attack').value === 'backdoor';
  $('source-control').hidden = !targeted;
  $('target-control').hidden = !targetRequired;
  $('alpha-control').hidden = !blended;
  $('status').textContent = $('dataset').value === 'imdb'
    ? 'IMDB uses MiniLM; text attacks add labels or a phrase trigger.'
    : 'Ready';
}
$('dataset').onchange = updateAttackOptions;
$('attack').onchange = updateAttackOptions;
$('scope').onchange = () => {
  const fullTraining = $('scope').value === 'full';
  $('sample-control').hidden = fullTraining;
};
updateAttackOptions();
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
  $('status').textContent = 'Starting extraction...';
  $('progress').hidden = false; $('progress-copy').hidden = false;
  $('extract').disabled = true;
  try {
    const response = await fetch('/api/extract', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset: $('dataset').value,
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
    const data = await waitForJob(job.job_id);
    const representations = data.representations || [data];
    $('summary').textContent = representations
      .map(item => `${item.encoder || 'minilm'}: ${item.samples} samples · ${item.feature_dim}D raw · ${item.reduced_dim}D PCA`)
      .join(' | ');
    $('artifacts').textContent = data.feature_file
      ? `Saved for reuse: ${representations.map(item => `${item.feature_file}${item.image_file ? ` · images: ${item.image_file}` : ''}`).join(' | ')}`
      : '';
    $('result').hidden = false; draw(data.visual_features, data.labels);
    $('status').textContent = data.poisoned === null ? 'Extraction complete' : `Extraction complete · ${data.poisoned} poisoned samples`;
  } catch (error) { $('status').textContent = `Error: ${error.message}`; }
  finally { $('extract').disabled = false; }
};

async function waitForJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok || job.status === 'error') throw new Error(job.message || 'Extraction failed');
    $('progress').value = job.progress;
    $('progress-copy').textContent = `${job.progress}% · ${job.message}`;
    if (job.status === 'complete') return job.result;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}
