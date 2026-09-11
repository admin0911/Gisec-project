"""Leila: evaluate the verified CIFAR demo patch on held-out test images."""
from pathlib import Path
import json
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import small_cnn


def patch_specification(manifest, images, clean_reference):
    """Leila: verify each image trigger independently, including mixed attack groups."""
    path = Path(manifest['source_images'])
    if manifest.get('dataset', 'cifar10') not in ('cifar10', 'mnist'):
        return None
    match = re.search(r'-(backdoor|blended_injection|mixed_noise|mixed_all)-', path.name)
    if not match:
        return None
    attack = match[1]
    metadata = path.with_name(path.name.replace('-images.npz', '-evaluation.npz'))
    if not metadata.exists():
        metadata = path.with_name(path.name.replace('-images.npz', '-features.npz'))
    with np.load(metadata, allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], images.sample_ids):
            raise ValueError('Trigger metadata IDs do not match source images.')
        if 'labels' in saved and not np.array_equal(saved['labels'], images.labels):
            raise ValueError('Trigger metadata labels do not match source images.')
        poisoned = saved['is_poisoned'].astype(bool)
        types = saved['poison_type'].astype(str)
    n = len(images.images)
    if poisoned.shape != (n,) or types.shape != (n,) or not np.array_equal(poisoned, types != 'clean'):
        raise ValueError('Invalid attack metadata alignment or poison mask.')
    kinds = {'backdoor': ['backdoor'], 'blended_injection': ['blended_injection'],
             'mixed_noise': ['blended_injection'], 'mixed_all': ['backdoor', 'blended_injection']}[attack]
    allowed = set(kinds) | ({'label_flip'} if attack.startswith('mixed_') else set()) | {'clean'}
    if not set(types).issubset(allowed):
        raise ValueError('Attack types do not match the saved scenario.')
    specs = []
    for kind in kinds:
        rows = np.flatnonzero(types == kind)
        if not len(rows):
            raise ValueError('ASR requires recorded rows for each trigger.')
        targets = np.unique(images.labels[rows])
        if len(targets) != 1 or not 0 <= targets[0] < 10:
            raise ValueError('Trigger target cannot be determined unambiguously.')
        spec = dict(type='image_patch', target_label=int(targets[0]), size=3,
                    position='bottom_right', value=1.0, source='verified saved pixels and attack metadata')
        if kind == 'blended_injection':
            recipe = re.search(r'-a([0-9.]+)-t(\d+)-.*-seed(\d+)-images\.npz$', path.name)
            if not recipe:
                raise ValueError('Missing saved blend recipe; ASR cannot guess the trigger.')
            alpha, target, seed = float(recipe[1]), int(recipe[2]), int(recipe[3])
            if not 0 < alpha <= 1 or target != targets[0]:
                raise ValueError('Blend target or strength does not match metadata.')
            spec = dict(type=kind, target_label=target, alpha=alpha, noise_seed=seed+104729,
                        attack_seed=seed, source='saved recipe verified against attacked pixels')
        for index in rows:
            expected = apply_image_trigger(clean_reference[int(index)][0].float(), spec)
            if not np.array_equal(expected.numpy(), images.images[index]):
                raise ValueError('Saved pixels do not match the trigger recipe; ASR stopped.')
        specs.append(spec)
    if attack.startswith('mixed_'):
        return dict(type='mixed', triggers=specs, aggregation='unweighted mean of separately applied trigger ASRs; label flips excluded')
    return specs[0]


def evaluate_patch_model(model, test, spec, ordinary, batch_size=128):
    """Use original non-target test labels as the ASR denominator; never modify test data."""
    target = spec['target_label']
    predictions, labels, positions = [], [], []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for x, y in DataLoader(test.dataset, batch_size=batch_size, shuffle=False):
            eligible = y != target
            selected = torch.nonzero(eligible).flatten()
            if len(selected):
                patched = x[eligible].clone()
                patched = apply_image_trigger(patched, spec)
                logits = model(patched)
                if logits.shape != (len(patched), 10) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid patch evaluation logits.')
                predictions.extend(logits.argmax(1).cpu().tolist())
                labels.extend(y[eligible].tolist())
                positions.extend((selected + offset).tolist())
            offset += len(y)
    if not positions:
        raise ValueError('Patch ASR requires non-target test samples.')
    positions = np.asarray(positions)
    pred, labels = np.asarray(predictions), np.asarray(labels)
    clean_pred = np.asarray(ordinary)[positions]
    correct = clean_pred == labels
    metrics = dict(non_target_samples=len(pred), target_label=target,
                   untriggered_target_rate=float(np.mean(clean_pred == target)),
                   asr_non_target=float(np.mean(pred == target)),
                   conditional_asr=float(np.mean(pred[correct] == target)) if correct.any() else None)
    return metrics, dict(sample_ids=test.sample_ids[positions], predictions=pred, labels=labels)


def add_patch_metrics(run, test, spec):
    if run['model_name'] not in ('small_cnn_v1', 'small_cnn_mnist_v1'):
        raise ValueError('Unsupported checkpoint architecture for CIFAR patch ASR.')
    with np.load(run['artifacts']['predictions'], allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], test.sample_ids):
            raise ValueError('Saved test prediction IDs do not match official test inputs.')
        actual = np.asarray([int(test.dataset[i][1]) for i in range(len(test.dataset))])
        if not np.array_equal(saved['labels'], actual):
            raise ValueError('Saved test labels differ from official labels.')
        ordinary = saved['predictions'].copy()
    model = small_cnn(channels=1 if run['model_name'] == 'small_cnn_mnist_v1' else 3)
    model.load_state_dict(torch.load(run['artifacts']['checkpoint'], map_location='cpu', weights_only=True))
    # Leila: mixed triggers are applied separately, never stacked onto the same test image.
    specs = spec['triggers'] if spec['type'] == 'mixed' else [spec]
    per_trigger = {}
    artifacts = {}
    for trigger in specs:
        metrics, predictions = evaluate_patch_model(model, test, trigger, ordinary)
        key = trigger['type']
        output = Path(run['artifacts']['report']).parent / (key + '_test_predictions.npz')
        np.savez_compressed(output, **predictions)
        per_trigger[key] = metrics
        artifacts[key] = str(output)
    run['backdoor_metrics'] = dict(next(iter(per_trigger.values())))
    if spec['type'] == 'mixed':
        for key in ('asr_non_target', 'untriggered_target_rate', 'conditional_asr'):
            values = [m[key] for m in per_trigger.values()]
            run['backdoor_metrics'][key] = float(np.mean(values)) if all(v is not None for v in values) else None
        for kind, metrics in per_trigger.items():
            for key, value in metrics.items():
                run['backdoor_metrics'][kind + '_' + key] = value
    run['trigger_metrics'] = per_trigger
    run['backdoor_trigger'] = spec
    run['artifacts']['triggered_predictions_by_type'] = artifacts
    run['artifacts']['triggered_predictions'] = next(iter(artifacts.values()))
    Path(run['artifacts']['report']).write_text(json.dumps(run, indent=2, allow_nan=False), encoding='utf-8')
    return run


# Leila: reconstruct the saved blend recipe and verify every poisoned training image.
def apply_image_trigger(pixels, spec):
    result = pixels.clone()
    if spec.get('type') == 'blended_injection':
        noise = np.clip(np.random.default_rng(spec['noise_seed']).normal(
            0.5, 0.2, size=tuple(result.shape[-3:])), 0, 1).astype(np.float32)
        trigger = torch.from_numpy(noise).to(device=result.device, dtype=result.dtype)
        return ((1-spec['alpha'])*result + spec['alpha']*trigger).clamp(0, 1)
    result[..., -spec['size']:, -spec['size']:] = spec['value']
    return result


def blend_specification(manifest, images, clean_reference):
    return patch_specification(manifest, images, clean_reference)
