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
    if (data && data.detectors && !Array.isArray(data.detectors) && data.detectors.blended_injection) {
      renderPipeline(data);
      return;
    }
    if (data && data.detectors && !Array.isArray(data.detectors) && data.detectors.backdoor_feature) {
      renderStandaloneBackdoorFeature(data);
      return;
    }
    // Leila: describe the single pixel representation separately from the CIFAR dual encoders.
    const mnist = data.dataset === 'mnist';
    const imdb = data.dataset === 'imdb';
    const single = mnist || imdb;

    // Leila: describe the available scan stages without encoder implementation details.
    get('scan-description').textContent = imdb ? 'Scanning for label flipping in review text.' : 'Scanning for label flipping, suspicious patches and blended noise.';
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
    renderBackdoorFeature(data.backdoor_feature);
    get('not-flagged').textContent = number(data.summary.not_flagged);
    get('uncertain').textContent = number(data.summary.uncertain);
    get('suspected').textContent = number(data.summary.suspected_label_flip);
    const total = Math.max(1, data.samples);
    requestAnimationFrame(() => {
      get('bar-clean').style.width = `${data.summary.not_flagged / total * 100}%`;
      get('bar-uncertain').style.width = `${data.summary.uncertain / total * 100}%`;
      get('bar-suspected').style.width = `${data.summary.suspected_label_flip / total * 100}%`;
    });
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
    // Leila: always show the noise stage, including missing historical results and text applicability.
    if (get('blended-results')) {
      const blend=data.blended_scan;
      get('blended-results').hidden=false;
      get('blended-counts').hidden=!blend || !blend.applicable;
      get('blended-status').hidden=false;
      get('blended-status').textContent=imdb ? 'Not applicable: image-noise detection requires pixels. IMDB uses label and repeated-phrase checks.' : 'Not run in this saved scan. Run Scan dataset again to include noise detection.';
      // Leila: the moved details are independent of the findings section.
      if (get('blended-details')) get('blended-details').hidden=!blend;
      get('blended-availability').textContent=blend?'Stage 3 · Blended-injection checks: '+(blend.applicable?'Completed':'Inconclusive'):'Blended-injection check not run for this saved scan.';
      if(blend) {
        get('blended-counts').hidden=!blend.applicable;
        // Leila: only show the status paragraph when this check is inconclusive.
        // Leila: completed checks are represented by the result cards alone.
        get('blended-status').hidden=blend.applicable;
        get('blended-status').textContent=blend.applicable?'':(blend.status || 'This check could not assess this input.');
        get('blended-not-flagged').textContent=number(data.samples-blend.flagged);
        get('blended-flagged').textContent=number(blend.flagged);
        get('review-blended').href=`${reviewUrl}&group=uncertain`;
        get('blended-settings').textContent=`Method: ${blend.method || "legacy consensus pixels"}. Scores indicate suspicion, not poisoning probability. Settings: ${JSON.stringify(blend.settings)}. Evidence: ${JSON.stringify(blend.evidence || {})}`;
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

      const e=data.demo_evaluation, pct=v=>v==null?'N/A (no poison in this dataset)':`${(v*100).toFixed(2)}%`;
      get('imdb-evaluation').hidden=false;
      get('imdb-evaluation').textContent=e ? `Demo evaluation · 2 of 3 rule · Precision ${pct(e.precision)} · Recall ${pct(e.recall)} · ${number(e.caught)} of ${number(e.known_poisoned)} poisoned reviews caught · ${number(e.false_positives)} clean reviews flagged. Known identities are used only for evaluation.` : 'Demo evaluation unavailable: no known poison metadata.';
    }
  }
  function renderPipeline(data) {
    const detector = data.detectors.blended_injection;
    const evidence = detector.evidence || {};
    const flags = detector.flags || [];
    const flagged = flags.filter(Boolean).length;
    get('scan-description').textContent = `${data.samples.toLocaleString()} CIFAR-10 samples processed through the detector pipeline.`;
    get('scan-summary').textContent = 'Extraction complete · applicable detectors complete';
    get('blended-pipeline-summary').textContent = evidence.target_class === null || evidence.target_class === undefined
      ? 'No blended-injection signature identified.'
      : `Strong blended-injection signal detected in target class ${evidence.target_class} · ${flagged.toLocaleString()} samples flagged for review.`;
    get('pipeline-target').textContent = evidence.target_class ?? 'None';
    get('pipeline-flagged').textContent = flagged.toLocaleString();
    get('pipeline-contrast').textContent = Number(evidence.contrast || 0).toFixed(2);
    get('pipeline-rate').textContent = `${(flagged / Math.max(1, data.samples) * 100).toFixed(2)}%`;
    get('pipeline-ratio').textContent = Number(evidence.class_ratio || 0).toFixed(2);
    const contrasts = Object.entries(evidence.class_contrasts || {});
    const strongest = contrasts.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
    get('pipeline-strongest').textContent = strongest ? strongest[0] : 'None';
    drawPipelineChart(contrasts);
    const examples = get('pipeline-examples');
    examples.replaceChildren();
    (detector.top_samples || []).forEach(sample => {
      const card = document.createElement('figure');
      card.className = 'pipeline-example';
      const image = document.createElement('img');
      image.src = sample.image;
      image.alt = `Top flagged candidate ${sample.sample_id}`;
      const caption = document.createElement('figcaption');
      caption.textContent = `${sample.sample_id} · score ${Number(sample.score).toFixed(3)}`;
      card.append(image, caption);
      examples.appendChild(card);
    });
    if (!detector.top_samples?.length) examples.textContent = 'No samples were flagged.';
    get('blended-pipeline-results').hidden = false;
    get('scan-result').querySelector('.visual-summary').hidden = true;
    get('review-uncertain').closest('.assessment-grid').hidden = true;
    get('patch-results').hidden = true;
    get('scan-result').querySelectorAll(':scope > details').forEach(details => { details.hidden = true; });
    const reviewJob = data.review_job_id || jobId;
    const reviewUrl = `/review?job=${encodeURIComponent(reviewJob)}`;
    get('human-review').href = reviewUrl;
    get('human-review').hidden = false;
    get('prepare-training').href = `/train?scan=${encodeURIComponent(reviewJob)}`;
    get('prepare-training').hidden = false;
    get('review-saved-status').hidden = false;
    loadReviewSummaryFor(reviewJob);
    get('scan-result').hidden = false;
  }
  function renderBackdoorFeature(track) {
    if (!track) { get('backdoor-feature-results').hidden = true; return; }
    const candidates = track.candidate_flags || [];
    const agreement = track.agreement_flags || [];
    const detectorNames = Object.keys(track.detectors || {});
    get('backdoor-feature-summary').textContent =
      `${detectorNames.length} feature checks ran across ResNet-18 and DINOv2 representations.`;
    get('backdoor-candidates').textContent = number(candidates.filter(Boolean).length);
    get('backdoor-agreement').textContent = number(agreement.filter(Boolean).length);
    get('backdoor-detectors').textContent = number(detectorNames.length);
    drawVoteChart(track.flag_count || []);
    get('backdoor-feature-results').hidden = false;
  }
  function renderStandaloneBackdoorFeature(data) {
    const track = data.detectors.backdoor_feature;
    const candidates = track.candidate_flags || [];
    const agreement = track.agreement_flags || [];
    const detectorNames = Object.keys(track.detectors || {});
    get('scan-description').textContent = `${number(data.samples)} ${data.dataset || 'dataset'} samples processed through feature backdoor detectors.`;
    get('scan-summary').textContent = 'Feature backdoor scan complete';
    get('backdoor-feature-summary').textContent = 'Feature detectors are shown alongside the existing backdoor safeguards.';
    get('backdoor-candidates').textContent = number(candidates.filter(Boolean).length);
    get('backdoor-agreement').textContent = number(agreement.filter(Boolean).length);
    get('backdoor-detectors').textContent = number(detectorNames.length);
    drawVoteChart(track.flag_count || []);
    get('backdoor-feature-results').hidden = false;
    get('scan-result').querySelector('.visual-summary').hidden = true;
    get('review-uncertain').closest('.assessment-grid').hidden = true;
    get('patch-results').hidden = true;
    get('human-review').hidden = true;
    get('prepare-training').hidden = true;
    get('scan-result').hidden = false;
  }
  function drawVoteChart(votes) {
    const canvas = get('backdoor-votes-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const counts = [votes.filter(value => value === 0).length, votes.filter(value => value === 1).length, votes.filter(value => value >= 2).length];
    const max = Math.max(1, ...counts);
    const started = performance.now();
    function frame(now) {
      const phase = Math.min(1, (now - started) / 700);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ['0 votes', '1 vote', '2 votes'].forEach((label, index) => {
        const height = counts[index] / max * (canvas.height - 50) * phase;
        const x = 90 + index * 250;
        ctx.fillStyle = index === 2 ? '#d45c67' : index === 1 ? '#d99421' : '#2b956d';
        ctx.fillRect(x, canvas.height - 30 - height, 120, height);
        ctx.fillStyle = '#53637b'; ctx.font = '14px Segoe UI'; ctx.fillText(label, x + 25, canvas.height - 10);
      });
      if (phase < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function drawPipelineChart(entries) {
    const canvas = get('pipeline-contrast-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const max = Math.max(1, ...entries.map(([, value]) => Number(value)));
    const started = performance.now();
    function frame(now) {
      const phase = Math.min(1, (now - started) / 800);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const width = entries.length ? (canvas.width - 80) / entries.length : canvas.width;
      entries.forEach(([label, value], index) => {
        const height = Number(value) / max * (canvas.height - 55) * phase;
        const x = 40 + index * width + width * .18;
        ctx.fillStyle = String(label) === String(get('pipeline-target').textContent) ? '#d45c67' : '#2b956d';
        ctx.fillRect(x, canvas.height - 30 - height, width * .64, height);
        ctx.fillStyle = '#53637b'; ctx.font = '12px Segoe UI';
        ctx.fillText(label, x + width * .2, canvas.height - 10);
      });
      if (phase < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function loadReviewSummaryFor(reviewJob) {
    fetch('/api/review/summary', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id:reviewJob})})
      .then(response => response.json().then(data => ({response, data})))
      .then(({response, data}) => {
        if (!response.ok) throw new Error(data.error || 'Could not load saved reviews.');
        get('review-saved-status').textContent = data.saved
          ? `Saved human review · ${number(data.saved)} choices saved · ${number(data.unreviewed)} not yet reviewed`
          : 'No human review decisions saved yet.';
        for (const name of ['keep','quarantine','unsure']) get(`saved-${name}`).textContent = number(data[name]);
        get('review-saved-counts').hidden = !data.saved;
      })
      .catch(error => { get('review-saved-status').textContent = `Saved review summary unavailable: ${error.message}`; });
  }
  async function poll() {
    if (!jobId) { get('scan-status').textContent = 'Build a blended-injection dataset for unified results, or start a label-flip scan from Prepare dataset.'; get('scan-progress').hidden = true; return; }
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error('Scan unavailable. The server may have restarted; start a new scan from Prepare dataset.');
      const job = await response.json();
      // Leila: show the selected dataset for both active and restored scans.
      const info = job.result?.dataset_info || job.dataset_info;
      if (info && get('dataset-information')) { get('dataset-information').hidden=false; get('dataset-information-text').textContent=info.description; }
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
