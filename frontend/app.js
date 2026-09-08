const $ = id => document.getElementById(id);
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
      body: JSON.stringify({dataset: $('dataset').value, limit: Number($('limit').value), attack: $('attack').value})
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Extraction failed');
    const data = await waitForJob(job.job_id);
    $('summary').textContent = `${data.dataset}: ${data.samples} samples · ${data.feature_dim}D raw · ${data.reduced_dim}D PCA`;
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
