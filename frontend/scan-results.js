(() => {
  const get = id => document.getElementById(id);
  // Leila: preserve this scan in its history entry while displaying the short /scan address.
  const params = new URLSearchParams(location.search);
  let savedJob = null;
  try { savedJob = sessionStorage.getItem('label-flip-job'); } catch (_) {}
  const jobId = params.has('job') ? params.get('job') : (history.state?.poisonGuardScan?.job || savedJob);
  history.replaceState({...(history.state || {}),poisonGuardScan:{job:jobId}},'', '/scan');
  if (jobId) { try { sessionStorage.setItem('label-flip-job',jobId); } catch (_) {} }
  const names = {knn:'kNN label disagreement', class_distance:'Class distance', confident_learning:'Confident Learning',
    minilm:'MiniLM', pixels:'MNIST pixels', resnet18:'ResNet18', dinov2_vits14:'DINOv2'};
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
    // Leila: describe the single pixel representation separately from the CIFAR dual encoders.
    const mnist = data.dataset === 'mnist';
    const imdb = data.dataset === 'imdb';
    const single = mnist || imdb;

    // Leila: describe the available scan stages without encoder implementation details.
    get('scan-description').textContent = imdb ? 'Scanning for label flipping in review text.' : 'Scanning for label flipping and suspicious repeated bright patches.';
    get('clean-explanation').textContent = single ? `No detector flagged these ${imdb ? 'reviews' : 'images'}. This does not guarantee clean data.` : 'Not flagged by either encoder’s combined rule. This does not guarantee clean data.';
    get('uncertain-explanation').textContent = single ? 'One of the three detectors flagged these samples.' : 'The encoders disagree. Only one reaches two detector votes.';
    get('suspected-explanation').textContent = single ? 'At least two of three detectors flagged these samples. Suspected, not confirmed.' : 'Suspected poisoning, not confirmed. Both encoders reach at least two detector votes.';
    // Leila: MNIST opens the shared preparation design with training disabled.
    get('prepare-training').hidden = false;
    get('human-review').hidden = false;
    get('review-saved-status').hidden = false;
    get('prepare-training').target = single ? '_blank' : '_self';
    get('prepare-training').rel = 'noopener';
    // Leila: link completed scans to optional review while retaining original detector counts.
    // Leila: the review controller keeps context and shortens this link to /review.
    const reviewUrl = `/review?job=${encodeURIComponent(jobId)}`;
    get('human-review').href = reviewUrl;
    // Leila: include patch-only findings in the same review flow.
    get('review-patches').href = `${reviewUrl}&group=uncertain`;
    // Leila: prepare from this scan and its saved review choices, without starting training.
    get('prepare-training').href = `/train?scan=${encodeURIComponent(jobId)}${imdb ? '&dataset=imdb' : mnist ? '&dataset=mnist' : ''}`;
    get('review-uncertain').href = `${reviewUrl}&group=uncertain`;
    get('review-suspected').href = `${reviewUrl}&group=suspected_label_flip`;
    get('scan-summary').textContent = `${number(data.samples)} ${imdb ? 'IMDB' : mnist ? 'MNIST' : 'CIFAR-10'} samples scanned · Experimental assessment`;
    get('not-flagged').textContent = number(data.summary.not_flagged);
    get('uncertain').textContent = number(data.summary.uncertain);
    get('suspected').textContent = number(data.summary.suspected_label_flip);
    for (const row of data.detectors) {
      const tr = document.createElement('tr');
      for (const value of [names[row.encoder], names[row.detector], number(row.flagged), `${(row.rate*100).toFixed(2)}%`, row.threshold_label || `> ${row.threshold.toFixed(4)}`]) {
        const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
      }
      get('detector-rows').appendChild(tr);
    }
    for (const sample of data.examples) {
      const p = document.createElement('p');
      p.textContent = `${sample.sample_id} · supplied label ${sample.label} · ${sample.assessment.replaceAll('_',' ')} · ${imdb ? `MiniLM ${sample.text_votes}/3` : mnist ? `Pixels ${sample.pixel_votes}/3` : `ResNet18 ${sample.resnet_votes}/3 · DINOv2 ${sample.dino_votes}/3`}`;
      get('scan-examples').appendChild(p);
    }
    if (!data.examples.length) get('scan-examples').textContent = 'No samples reached the combined review rule.';
    // Leila: patch evidence is not a fourth vote in the label-flip assessment.
    // Leila: display the MNIST-only connector result independently of label voting.
    if (get('blended-results')) {
      const blend=data.blended_scan;
      get('blended-results').hidden=!blend;
      // Leila: the moved details are independent of the findings section.
      if (get('blended-details')) get('blended-details').hidden=!blend;
      get('blended-availability').textContent=blend?'Stage 3 · Blended-injection checks: '+(blend.applicable?'Completed':'Inconclusive'):'Blended-injection check not run for this saved scan.';
      if(blend) {
        get('blended-counts').hidden=!blend.applicable;
        // Leila: only show the status paragraph when this check is inconclusive.
        get('blended-status').hidden=blend.applicable;
        get('blended-status').textContent=blend.applicable?'':'Inconclusive: too few consensus pixels. This check cannot assess this input.';
        get('blended-not-flagged').textContent=number(data.samples-blend.flagged);
        get('blended-flagged').textContent=number(blend.flagged);
        get('review-blended').href=`${reviewUrl}&group=uncertain`;
        get('blended-settings').textContent=`Consensus pixels: ${blend.consensus_pixel_count}. Score is the fraction disturbed, not a probability. Consensus: ${blend.settings.consensus}; tolerance: ${blend.settings.tolerance}; flag when score > ${blend.settings.flag_fraction}.`;
        get('blended-examples').replaceChildren();
        for(const sample of blend.examples) {const p=document.createElement('p');p.textContent=`${sample.sample_id} · label ${sample.label} · score ${sample.score.toFixed(3)}`;get('blended-examples').appendChild(p);}
      }
    }
    const patch = data.patch_scan;
    get('patch-results').hidden = !patch;
    // Leila: the relocated details remain hidden for text and older scans without patches.
    get('patch-details').hidden = !patch;
    if (patch) {
      // Leila: presentation only; preserve the detector's existing binary review flags.
      get('patch-not-flagged').textContent = number(data.samples - patch.flagged);
      get('patch-needs-review').textContent = number(patch.flagged);
      for (const pattern of patch.patterns) {
        const p = document.createElement('p');
        p.textContent = `${pattern.size} × ${pattern.size} patch · row ${pattern.row+1}, column ${pattern.column+1} · label ${pattern.dominant_label} · ${number(pattern.support)} matches · ${(pattern.purity*100).toFixed(1)}% label agreement`;
        get('patch-patterns').appendChild(p);
      }
      if (!patch.patterns.length) get('patch-patterns').textContent = 'No repeated patches met the detector thresholds.';
      for (const sample of patch.examples) {
        const p = document.createElement('p');
        p.textContent = `${sample.sample_id} · supplied label ${sample.label} · patch score ${sample.score.toFixed(3)}`;
        get('patch-examples').appendChild(p);
      }
    }
    get('scan-result').hidden = false;
    loadReviewSummary();
    // Leila: phrase flags request review; no automatic confirmed-poison verdict.
    if (get('phrase-results')) {
      get('phrase-results').hidden = !data.phrase_scan;
      get('phrase-details').hidden = !data.phrase_scan;
      if (data.phrase_scan) {
        get('scan-description').textContent='Scanning for label flipping and suspicious repeated phrases.';
        // Leila: use the same counts and review navigation as image backdoor cards.
        get('phrase-not-flagged').textContent=number(data.samples-data.phrase_scan.flagged);
        get('phrase-needs-review').textContent=number(data.phrase_scan.flagged);
        get('review-phrases').href=`${reviewUrl}&group=uncertain`;
        get('phrase-patterns').textContent='';
        for (const pattern of data.phrase_scan.patterns) {
          const p=document.createElement('p');p.textContent=`“${pattern.phrase}” · ${number(pattern.support)} matches · ${(pattern.purity*100).toFixed(1)}% label agreement`;
          get('phrase-patterns').appendChild(p);
        }
      }
    }
    if (data.training_enabled === false && !imdb) {
      get('prepare-training').removeAttribute('href');
      get('prepare-training').textContent='Training integration pending';
      get('prepare-training').setAttribute('aria-disabled','true');
    }
    if (imdb) {

      const e=data.demo_evaluation, pct=v=>v==null?'N/A':`${(v*100).toFixed(2)}%`;
      get('imdb-evaluation').hidden=false;
      get('imdb-evaluation').textContent=e ? `Demo evaluation · 2 of 3 rule · Precision ${pct(e.precision)} · Recall ${pct(e.recall)} · ${number(e.caught)} of ${number(e.known_poisoned)} poisoned reviews caught · ${number(e.false_positives)} clean reviews flagged. Known identities are used only for evaluation.` : 'Demo evaluation unavailable: no known poison metadata.';
    }
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
