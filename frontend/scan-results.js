(() => {
  const get = id => document.getElementById(id);
  const jobId = new URLSearchParams(location.search).get('job') || sessionStorage.getItem('label-flip-job');
  const names = {knn:'kNN label disagreement', class_distance:'Class distance', confident_learning:'Confident Learning',
    resnet18:'ResNet18', dinov2_vits14:'DINOv2'};
  const number = n => n.toLocaleString();
  // Leila: load human decisions independently so the original detector assessment remains intact.
  let reviewRequest = 0;
  async function loadReviewSummary() {
    const request = ++reviewRequest;
    try {
      const response = await fetch('/api/review/summary', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id:jobId})});
      const data = await response.json();
      if (request !== reviewRequest) return;
      if (!response.ok) throw new Error(data.error || 'Could not load saved reviews.');
      get('review-saved-status').textContent = data.saved
        ? `Saved human review · ${number(data.saved)} choices saved · ${number(data.unreviewed)} not yet reviewed`
        : 'No human review decisions saved yet.';
      for (const name of ['keep','quarantine','unsure']) get(`saved-${name}`).textContent = number(data[name]);
      get('review-saved-counts').hidden = !data.saved;
    } catch (error) {
      if (request !== reviewRequest) return;
      get('review-saved-counts').hidden = true;
      get('review-saved-status').textContent = `Saved review summary unavailable: ${error.message}`;
    }
  }
  function render(data) {
    // Leila: link completed scans to optional review while retaining original detector counts.
    const reviewUrl = `/human-review.html?job=${encodeURIComponent(jobId)}`;
    get('human-review').href = reviewUrl;
    // Leila: prepare from this scan and its saved review choices, without starting training.
    get('prepare-training').href = `/training.html?scan=${encodeURIComponent(jobId)}`;
    get('review-uncertain').href = `${reviewUrl}&group=uncertain`;
    get('review-suspected').href = `${reviewUrl}&group=suspected_label_flip`;
    get('scan-summary').textContent = `${number(data.samples)} CIFAR-10 samples scanned · Experimental assessment`;
    get('not-flagged').textContent = number(data.summary.not_flagged);
    get('uncertain').textContent = number(data.summary.uncertain);
    get('suspected').textContent = number(data.summary.suspected_label_flip);
    for (const row of data.detectors) {
      const tr = document.createElement('tr');
      for (const value of [names[row.encoder], names[row.detector], number(row.flagged), `${(row.rate*100).toFixed(2)}%`, row.threshold.toFixed(4)]) {
        const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
      }
      get('detector-rows').appendChild(tr);
    }
    for (const sample of data.examples) {
      const p = document.createElement('p');
      p.textContent = `${sample.sample_id} · supplied label ${sample.label} · ${sample.assessment.replaceAll('_',' ')} · ResNet18 ${sample.resnet_votes}/3 · DINOv2 ${sample.dino_votes}/3`;
      get('scan-examples').appendChild(p);
    }
    if (!data.examples.length) get('scan-examples').textContent = 'No samples reached the combined review rule.';
    get('scan-profile').textContent = data.profile;
    get('scan-limitation').textContent = data.limitation;
    get('scan-file').textContent = `Saved results: ${data.result_file}`;
    get('scan-result').hidden = false;
    loadReviewSummary();
  }
  async function poll() {
    if (!jobId) { get('scan-status').textContent = 'Start a label-flip scan from Prepare dataset.'; get('scan-progress').hidden = true; return; }
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error('Scan unavailable. The server may have restarted; start a new scan from Prepare dataset.');
      const job = await response.json();
      get('scan-status').textContent = job.message;
      get('scan-progress').value = job.progress;
      if (job.status === 'error') throw new Error(job.message);
      if (job.status === 'complete') { render(job.result); return; }
      setTimeout(poll, 1000);
    } catch (error) {
      get('scan-status').textContent = `Scan could not finish: ${error.message}`;
      get('scan-progress').hidden = true;
    }
  }
  // Leila: browser Back may restore a cached page; refresh only its saved review summary.
  window.addEventListener('pageshow', event => {
    if (event.persisted && !get('scan-result').hidden) loadReviewSummary();
  });
  poll();
})();
