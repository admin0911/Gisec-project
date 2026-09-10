"""MNIST web adapter using the saved image connector, never encoder vectors."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from poison_features import ImageInputBundle
from experiments.scan_mnist_pixels import pixel_input, scan_pixels
from detectors.output_connector import to_jsonable
from .scan_cache import scan_identity, save_cache_record


def profile():
    return dict(name='mnist_pixels_19_of_20_v1', dataset='mnist', k=20,
        knn_threshold=.95, class_threshold=.1, folds=5, diagnostic_seed=2026,
        limitation='MNIST normalized pixels; provisional thresholds, not independently calibrated. '
        '0 votes: not flagged; 1 vote: needs review; 2 or 3 votes: suspected label flip. '
        'Flags are review evidence, not proof of poisoning. No CIFAR thresholds or encoder vectors are used.')


def run(feature_path, output_dir, progress):
    feature_path = Path(feature_path)
    config = profile()
    identity = scan_identity((feature_path,), config)
    images = ImageInputBundle.load(feature_path.with_name(feature_path.name.replace('-features.npz', '-images.npz')))
    # Read only public row identifiers and supplied labels; never load poison truth or embeddings.
    with np.load(feature_path, allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], images.sample_ids) or not np.array_equal(saved['labels'], images.labels):
            raise ValueError('MNIST image IDs and labels do not match the selected extraction.')
    inputs = pixel_input(images)
    if (inputs.y.dtype.kind not in 'iu' or np.any((inputs.y < 0) | (inputs.y > 9))
            or any(not str(sid).startswith('mnist-train:') for sid in inputs.sample_ids)):
        raise ValueError('Expected MNIST training IDs and digit labels 0–9.')
    _, counts = np.unique(inputs.y, return_counts=True)
    if len(inputs.y) <= 20 or len(counts) < 2 or counts.min() < 5:
        raise ValueError('MNIST scan needs at least 21 images and five samples per supplied class.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    step = 0
    detector_name = "Starting"
    def report(message):
        nonlocal step, detector_name
        if ' of 3:' in message:
            local_step, detector_name = message.split(' of 3:', 1)
            step = int(local_step)
            message = ''
        # Leila: fold progress belongs inside the current label detector.
        progress(step * 2, f'Stage 1 of 3 · Label-flip checks\nCheck {step} of 3 · {detector_name.strip()} — MNIST pixels' + (f'\n{message.strip()}' if message.strip() else ''))
    with threadpool_limits(limits=4):
        # Leila: apply the same threshold recorded in the scan/cache profile.
        results, timings = scan_pixels(inputs, knn_threshold=config['knn_threshold'], progress=report)
    votes = np.sum(np.stack([r['flags'] for r in results.values()]), axis=0)
    states = np.full(len(votes), 'not_flagged', dtype='<U24')
    states[votes == 1] = 'uncertain'
    states[votes >= 2] = 'suspected_label_flip'
    assessment = dict(sample_ids=inputs.sample_ids, assessment=states, flags=votes > 0,
        vote_counts={'pixels': votes}, summary={name:int(np.sum(states == name)) for name in
            ('not_flagged', 'uncertain', 'suspected_label_flip')}, settings=config)
    # Leila: scan the same original pixels after label checks, without changing label votes.
    from detectors.backdoor.web_patch import scan_patch
    patch_result, patch_ui = scan_patch(images, assessment, progress)
    # Leila: run MNIST background checks after patches through the same image connector.
    from detectors.blended_injection import ConsensusPixelDetector
    progress(6.95, 'Stage 3 of 3 · Blended-injection checks\nCheck 1 of 1 · Consensus pixels — MNIST')
    blended_result = ConsensusPixelDetector().analyze(images)
    evidence = blended_result['evidence']
    selected_blended = np.flatnonzero(blended_result['flags'])
    selected_blended = selected_blended[np.argsort(-blended_result['scores'][selected_blended], kind='stable')][:24]
    blended_ui = dict(applicable=evidence['applicable'],status=evidence['status'],
        flagged=int(blended_result['flags'].sum()),consensus_pixel_count=evidence['consensus_pixel_count'],
        settings=blended_result['settings'],examples=[dict(sample_id=str(images.sample_ids[i]),
            label=int(images.labels[i]),score=float(blended_result['scores'][i])) for i in selected_blended])
    rows = []
    for name, result in results.items():
        cutoff = '≥ 0.95 (19/20)' if name == 'knn' else '≥ 0.10' if name == 'class_distance' else 'Cleanlab pruning'
        rows.append(dict(encoder='pixels', detector=name, flagged=int(result['flags'].sum()),
            rate=float(result['flags'].mean()), threshold=None, threshold_label=cutoff))
        # Keep scores and flags in the shared output format without huge duplicate neighbour ID arrays.
        result['evidence'].pop('neighbour_sample_ids', None)
    selected = np.flatnonzero(votes > 0)[:24]
    examples = [dict(sample_id=str(inputs.sample_ids[i]),label=int(inputs.y[i]),
        assessment=str(states[i]),pixel_votes=int(votes[i])) for i in selected]
    ui = dict(blended_scan=blended_ui, patch_scan=patch_ui, dataset='mnist', samples=len(votes), summary=assessment['summary'], detectors=rows,
        examples=examples, profile=config['name'],limitation=config['limitation'],
        result_file=str(output/'results.json'),human_review_enabled=True,training_enabled=True)
    if scan_identity((feature_path,), config) != identity:
        raise ValueError('Inputs or settings changed during scanning. Rebuild and retry.')
    full = dict(dataset='mnist',profile=config,feature_files=[str(feature_path)],
        blended_scan=blended_result, patch_scan=patch_result, scans={'pixels':{'detectors':results}},assessment=assessment,ui_result=ui,detector_seconds=timings)
    (output/'results.json').write_text(json.dumps(to_jsonable(full),allow_nan=False),encoding='utf-8')
    save_cache_record(output,identity)
    return to_jsonable(ui)
