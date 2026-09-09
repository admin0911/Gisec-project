(() => {
  const $ = id => document.getElementById(id), params = new URLSearchParams(location.search);
  let version = params.get('version'), scan = params.get('scan'), active = false;
  let knownClean = false, comparisonData = null;
  const number = n => n.toLocaleString();
  async function api(route, data) {
    const response = await fetch(`/api/training/${route}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Training request failed.');
    return result;
  }
  function renderComparison(data) {
    comparisonData = data;
    $('comparison-rows').replaceChildren();
    // Leila: older saved comparisons have no reference; never invent its accuracy.
    for (const [key,title] of [['clean_reference','Clean reference'],['before_cleaning','Before cleaning'],['after_cleaning','After cleaning']]) {
      // Leila: hide the redundant row for verified clean demos, including saved runs.
      if (key === 'before_cleaning' && (data.before_cleaning_skipped || knownClean)) continue;
      const run = data.runs[key], row = document.createElement('tr');
      for (const value of [title,run ? `${(run.metrics.accuracy*100).toFixed(2)}%` : 'Not run']) {
        const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
      }
      $('comparison-rows').appendChild(row);
    }
    $('reference-status').textContent = data.runs.clean_reference ? '' : 'This saved run predates the clean reference. Start Train & compare to include it.';
    const base = data.before_cleaning_skipped || (knownClean && data.runs.clean_reference) ? 'clean_reference' : 'before_cleaning';
    const change = data.runs.after_cleaning.metrics.accuracy - data.runs[base].metrics.accuracy;
    $('accuracy-change').textContent = `Accuracy change versus ${base === 'clean_reference' ? 'clean reference' : 'before cleaning'}: ${change>=0?'+':''}${(change*100).toFixed(2)} percentage points`;
    $('comparison-note').textContent = data.limitation;
    $('comparison-file').textContent = `Saved: ${data.result_file}`;
    $('comparison').hidden = false;
  }
  async function poll(job) {
    active = true; $('train').disabled = true; $('epochs').disabled = true; $('training-progress').hidden = false;
    try {
      const data = await api('job',{job_id:job});
      if (data.version !== version) throw new Error('This training job belongs to a different prepared dataset.');
      $('training-status').textContent = data.message;
      $('training-progress').value = data.progress;
      if (data.status==='error') throw new Error(data.message);
      if (data.status==='complete') { renderComparison(data.result); active=false; $('train').disabled=false; $('epochs').disabled=false; return; }
      setTimeout(() => poll(job),1000);
    } catch(error) { $('training-status').textContent = `Training could not finish: ${error.message}`; active=false; $('train').disabled=false; $('epochs').disabled=false; }
  }
  // Leila: evaluation failure must not block training or alter the prepared rows.
  async function loadEvaluation() {
    try {
      const data = await api('evaluation',{version});
      if (!data.available) { $('evaluation-status').textContent=`Evaluation unavailable: ${data.reason}`; return; }
      knownClean = data.original.poisoned === 0;
      if (knownClean) $('training-description').textContent='For this clean input, compare two fresh models: clean reference and after cleaning. The duplicate before-cleaning run is skipped when its images and labels match the reference.';
      if (comparisonData && !active) renderComparison(comparisonData);
      $('evaluation-rows').replaceChildren();
      for (const [key,title] of [['original','Original input'],['kept','Kept for training'],['removed','Removed from training']]) {
        const counts = data[key], row = document.createElement('tr');
        for (const value of [title,number(counts.clean),number(counts.poisoned),number(counts.total)]) {
          const cell = document.createElement('td'); cell.textContent=value; row.appendChild(cell);
        }
        $('evaluation-rows').appendChild(row);
      }
      $('evaluation-status').textContent=''; $('evaluation-table').hidden=false;
      $('evaluation-note').textContent = data.original.poisoned === 0
        ? `This demo input has no injected poisoning. All ${number(data.removed.clean)} removed samples are known clean.`
        : `${number(data.removed.poisoned)} poisoned samples removed · ${number(data.kept.poisoned)} poisoned samples still kept · ${number(data.removed.clean)} clean samples removed.`;
    } catch(error) { $('evaluation-status').textContent=`Evaluation unavailable: ${error.message}`; }
  }
  $('train').addEventListener('click',async () => {
    if (active) return;
    active=true; comparisonData=null; $('train').disabled=true; $('comparison').hidden=true;
    try {
      const job = await api('start',{version,epochs:Number($('epochs').value)});
      params.set('run',job.job_id); history.replaceState(null,'',`?${params}`);
      poll(job.job_id);
    } catch(error) { $('training-status').textContent=error.message; active=false; $('train').disabled=false; }
  });
  async function init() {
    try {
      const data = version ? await api('prepared',{version}) : await api('prepare',{scan_id:scan});
      version=data.version; scan=data.scan_id; params.set('version',version); params.set('scan',scan);
      history.replaceState(null,'',`?${params}`);
      $('back-results').href=`/scan-results.html?job=${encodeURIComponent(scan)}`;
      // Leila: display exclusion totals from this version without changing its selection.
      $('kept').textContent=number(data.summary.kept);
      $('removed').textContent=number(data.summary.quarantined + data.summary.unresolved);
      $('removed-breakdown').textContent=`${number(data.summary.quarantined)} quarantined · ${number(data.summary.unresolved)} unresolved`;
      $('preparation-status').textContent=`Dataset ready · ${number(data.summary.total)} original samples · ${number(data.summary.kept)} kept`;
      $('prepared').hidden=false;
      loadEvaluation();
      if (params.get('run')) poll(params.get('run'));
    } catch(error) { $('preparation-status').textContent=`Could not prepare training data: ${error.message}`; }
  }
  init();
})();
