// Leila: navigation uses the short scan route with the selected job preserved.
(() => {
  const $ = id => document.getElementById(id), params = new URLSearchParams(location.search);
  // Leila: keep IDs in this history entry; session storage also supports reopening /train.
  let stored = history.state?.poisonGuardTraining;
  if (!stored) { try { stored = JSON.parse(sessionStorage.getItem('poison-guard-training') || 'null'); } catch (_) {} }
  const explicit = params.has('scan') || params.has('version');
  const context = explicit ? {scan:params.get('scan'),version:params.get('version'),run:params.get('run'),dataset:params.get('dataset')} : (stored || {});
  let version = context.version, scan = context.scan, runId = context.run, active = false;
  let mnist = context.dataset === 'mnist', imdb = context.dataset === 'imdb';
  function rememberTraining() {
    const selection = {version,scan,run:runId || null,dataset:context.dataset};
    history.replaceState({...(history.state || {}),poisonGuardTraining:selection},'','/train');
    try { sessionStorage.setItem('poison-guard-training',JSON.stringify(selection)); } catch (_) {}
  }
  let knownClean = false, comparisonData = null, selectionPolicy = null;
  let currentJob = null;
  const number = n => n.toLocaleString();
  async function api(route, data) {
    // Leila: distinguish a lost connection from a confirmed failed training run.
    let response;
    try {
      response = await fetch(`/api/training/${route}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    } catch(cause) {
      const error = new Error('Cannot reach the local server.'); error.retryable = true; throw error;
    }
    if (response.headers && !(response.headers.get('content-type') || '').includes('application/json')) throw new Error('Restart the local Python server, then refresh this page.');
    const result = await response.json();
    if (!response.ok) { const error = new Error(result.error || 'Training request failed.'); error.retryable = response.status >= 500; throw error; }
    return result;
  }
  // Leila: derive classification averages from saved confusion counts, including older runs.
  // Rows are true classes and columns are predictions; undefined class scores count as zero.
  function macroMetrics(metrics) {
    const matrix = metrics?.confusion_matrix;
    if (!Array.isArray(matrix) || !matrix.length || !matrix.every(row =>
      Array.isArray(row) && row.length === matrix.length && row.every(n => Number.isInteger(n) && n >= 0))) return null;
    const count = matrix.length;
    let precision = 0, f1 = 0, total = 0;
    for (let i = 0; i < count; i++) {
      const actual = matrix[i].reduce((sum,n) => sum+n,0);
      const predicted = matrix.reduce((sum,row) => sum+row[i],0);
      const correct = matrix[i][i];
      precision += predicted ? correct / predicted : 0;
      f1 += actual + predicted ? 2 * correct / (actual + predicted) : 0;
      total += actual;
    }
    return total ? {precision:precision/count,f1:f1/count} : null;
  }
  function renderComparison(data) {
    comparisonData = data;
    $('comparison-rows').replaceChildren();
    // Leila: older saved comparisons have no reference; never invent its accuracy.
    for (const [key,title] of [['clean_reference','Clean reference'],['before_cleaning','Before cleaning'],['after_cleaning','After cleaning']]) {
      // Leila: hide the redundant row for verified clean demos, including saved runs.
      if (key === 'before_cleaning' && (data.before_cleaning_skipped || knownClean)) continue;
      const run = data.runs[key], row = document.createElement('tr');
      // Leila: these describe image classification, separate from poison-removal precision.
      const macro = macroMetrics(run?.metrics);
      const percent = value => value == null ? 'N/A' : `${(value*100).toFixed(2)}%`;
      for (const value of [title,run ? percent(run.metrics.accuracy) : 'Not run',
        run ? percent(macro?.f1) : 'Not run',run ? percent(macro?.precision) : 'Not run']) {
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
  async function poll(job, failures = 0) {
    currentJob = job; $('check-training').hidden = true;
    active = true; $('train').disabled = true; $('epochs').disabled = true; $('training-progress').hidden = false;
    try {
      const data = await api('job',{job_id:job});
      if (data.version !== version) throw new Error('This training job belongs to a different prepared dataset.');
      $('training-status').textContent = data.message;
      $('training-progress').value = data.progress;
      if (data.status==='error') throw new Error(data.message);
      if (data.status==='complete') { renderComparison(data.result); active=false; $('train').disabled=false; $('epochs').disabled=false; return; }
      setTimeout(() => poll(job),1000);
    } catch(error) {
      if (error.retryable) {
        // Leila: retry only status reads; a connection error must not launch duplicate training.
        $('training-status').textContent = 'Connection lost. Training may still be running. Reconnecting…';
        if (failures < 4) setTimeout(() => poll(job,failures+1),Math.min(2000 * 2**failures,10000));
        else {
          $('training-status').textContent = 'Cannot reach the server. Keep it running, then check the existing training status.';
          $('check-training').hidden=false;
        }
        return;
      }
      $('training-status').textContent = `Training stopped: ${error.message}`;
      active=false; $('train').disabled=false; $('epochs').disabled=false;
    }
  }
  $('check-training').addEventListener('click',() => { if (currentJob) poll(currentJob); });
  // Leila: evaluation failure must not block training or alter the prepared rows.
  async function loadEvaluation() {
    try {
      const data = await api('evaluation',{version});
      if (!data.available) { $('evaluation-status').textContent=`Evaluation unavailable: ${data.reason}`; return; }
      knownClean = data.original.poisoned === 0;
      if (knownClean && !mnist && !imdb) $('training-description').textContent='For this clean input, compare two fresh models: clean reference and after cleaning. The duplicate before-cleaning run is skipped when its images and labels match the reference.';
      if (comparisonData && !active) renderComparison(comparisonData);
      if (mnist) { $('kept').textContent=number(data.kept.total); $('removed').textContent=number(data.removed.total); }
      $('evaluation-rows').replaceChildren();
      for (const [key,title] of [['original','Original input'],['kept','Kept for training'],['removed','Removed from training']]) {
        const counts = data[key], row = document.createElement('tr');
        for (const value of [title,number(counts.clean),number(counts.poisoned),number(counts.total)]) {
          const cell = document.createElement('td'); cell.textContent=value; row.appendChild(cell);
        }
        $('evaluation-rows').appendChild(row);
      }
      $('evaluation-status').textContent=''; $('evaluation-table').hidden=false;
      // Leila: removed = quarantined + unresolved, including saved human choices.
      const precision = data.removed.total ? data.removed.poisoned / data.removed.total : null;
      const recall = data.original.poisoned ? data.removed.poisoned / data.original.poisoned : null;
      const percent = value => value === null ? 'N/A' : `${(value*100).toFixed(2)}%`;
      $('evaluation-precision').textContent=percent(precision);
      $('evaluation-recall').textContent=percent(recall);
      $('evaluation-metrics').hidden=false;
      $('metric-note').textContent = [
        selectionPolicy?.endsWith('-2of3-v1') ? 'This older preparation kept unreviewed uncertain samples. Prepare a new version to apply the current policy.' : 'Needs review without a human decision is excluded as unresolved. Saved Keep decisions include samples; Quarantine and Unsure exclude them. Original data remains saved.',
        precision === null ? 'Precision is N/A because no samples were removed.' : '',
        recall === null ? 'Recall is N/A because the input contains no known poisoned samples.' : ''
      ].filter(Boolean).join(' ');
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
      runId = job.job_id; rememberTraining();
      poll(job.job_id);
    } catch(error) { $('training-status').textContent=error.message; active=false; $('train').disabled=false; }
  });
  async function init() {
    try {
      rememberTraining();
      if (!version && !scan) {
        $('preparation-status').textContent='Choose Prepare training dataset from a completed scan first.';
        $('back-results').href='/'; $('back-results').textContent='← Prepare dataset';
        return;
      }
      const data = version ? await api('prepared',{version}) : await api('prepare',{scan_id:scan});
      mnist = data.dataset === 'mnist' || mnist;
      imdb = data.dataset === 'imdb' || imdb;
      context.dataset = imdb ? 'imdb' : mnist ? 'mnist' : 'cifar10';
      selectionPolicy=data.policy_version;
      version=data.version; scan=data.scan_id; rememberTraining();
      // Leila: the same controls now train MNIST using its separate test split.
      if (mnist) {
        $('training-description').textContent='Compare fresh clean-reference, before-cleaning and after-cleaning models on the separate MNIST test set. An identical clean input skips the duplicate before-cleaning model.';
        $('classification-note').textContent='Image classification metrics on the separate clean MNIST test set; macro metrics average equally across the 10 digits.';
      }
      if (imdb) {
        $('training-description').textContent='Train fresh sentiment classifiers on frozen MiniLM features. Compare clean reference, before and after cleaning using the separate 25,000-review IMDB test set.';
        $('classification-note').textContent='Sentiment classification on the same separate IMDB test reviews. Macro metrics average across positive and negative classes.';
        $('training-runtime-note').textContent='First run extracts and saves test features. Later comparisons reuse them. Keep the server running.';
      }
      $('back-results').href=`/scan?job=${encodeURIComponent(scan)}`;
      // Leila: display exclusion totals from this version without changing its selection.
      $('kept').textContent=number(data.summary.kept);
      $('removed').textContent=number(data.summary.quarantined + data.summary.unresolved);
      $('removed-breakdown').textContent=`${number(data.summary.quarantined)} quarantined · ${number(data.summary.unresolved)} unresolved`;
      $('preparation-status').textContent=`Dataset ready · ${number(data.summary.total)} original samples · ${number(data.summary.kept)} kept`;
      $('prepared').hidden=false;
      loadEvaluation();
      if (runId) poll(runId);
    } catch(error) { $('preparation-status').textContent=`Could not prepare training data: ${error.message}`; }
  }
  init();
})();
