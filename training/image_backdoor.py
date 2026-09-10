"""Leila: evaluate the verified CIFAR demo patch on held-out test images."""
from pathlib import Path
import json
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import small_cnn


def patch_specification(manifest, images, clean_reference):
    path = Path(manifest['source_images'])
    if '-blended_injection-' in path.name and manifest.get('dataset', 'cifar10') == 'cifar10':
        return blend_specification(manifest, images, clean_reference)
    if manifest.get('dataset', 'cifar10') != 'cifar10' or '-backdoor-' not in path.name:
        return None
    feature = path.with_name(path.name.replace('-images.npz', '-features.npz'))
    with np.load(feature, allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], images.sample_ids) or not np.array_equal(saved['labels'], images.labels):
            raise ValueError('Patch evaluation metadata does not match the source images.')
        poisoned = saved['is_poisoned'].astype(bool)
        types = saved['poison_type']
    if poisoned.shape != (len(images.images),) or not poisoned.any():
        raise ValueError('Patch evaluation requires recorded poisoned rows.')
    if not np.all(types[poisoned] == 'backdoor'):
        raise ValueError('Unsupported mixed attack for patch ASR.')
    targets = np.unique(images.labels[poisoned])
    if len(targets) != 1 or not 0 <= targets[0] < 10:
        raise ValueError('Patch target cannot be determined unambiguously.')
    # Verify the historical fixed-patch recipe against actual pixels, not just its filename.
    for index in np.flatnonzero(poisoned):
        expected = clean_reference[int(index)][0].clone()
        expected[..., -3:, -3:] = 1.0
        if not np.array_equal(expected.numpy(), images.images[index]):
            raise ValueError('Saved attack differs from the supported white bottom-right 3x3 patch.')
    return dict(type='image_patch', target_label=int(targets[0]), size=3,
                position='bottom_right', value=1.0, source='verified saved pixels and attack metadata')


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
    if run['model_name'] != 'small_cnn_v1':
        raise ValueError('Unsupported checkpoint architecture for CIFAR patch ASR.')
    with np.load(run['artifacts']['predictions'], allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], test.sample_ids):
            raise ValueError('Saved test prediction IDs do not match official test inputs.')
        actual = np.asarray([int(test.dataset[i][1]) for i in range(len(test.dataset))])
        if not np.array_equal(saved['labels'], actual):
            raise ValueError('Saved test labels differ from official labels.')
        ordinary = saved['predictions'].copy()
    model = small_cnn()
    model.load_state_dict(torch.load(run['artifacts']['checkpoint'], map_location='cpu', weights_only=True))
    metrics, predictions = evaluate_patch_model(model, test, spec, ordinary)
    output = Path(run['artifacts']['report']).parent / ('blended_test_predictions.npz' if spec.get('type') == 'blended_injection' else 'patch_test_predictions.npz')
    np.savez_compressed(output, **predictions)
    run['backdoor_metrics'] = metrics
    run['backdoor_trigger'] = spec
    run['artifacts']['triggered_predictions'] = str(output)
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
    path = Path(manifest['source_images'])
    match = re.search(r'-blended_injection-a([0-9.]+)-t(\d+)-.*-seed(\d+)-images\.npz$', path.name)
    if not match:
        raise ValueError('Missing saved blend recipe; ASR cannot guess the trigger.')
    alpha, target, seed = float(match[1]), int(match[2]), int(match[3])
    if not 0 < alpha <= 1 or not 0 <= target < 10:
        raise ValueError('Invalid saved blend settings.')
    spec = dict(type='blended_injection', target_label=target, alpha=alpha,
                noise_seed=seed+104729, attack_seed=seed,
                source='saved recipe verified against every poisoned training image')
    feature = path.with_name(path.name.replace('-images.npz', '-features.npz'))
    with np.load(feature, allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_ids'], images.sample_ids) or not np.array_equal(saved['labels'], images.labels):
            raise ValueError('Blend metadata does not match source images.')
        poisoned = saved['is_poisoned'].astype(bool)
        types = saved['poison_type'].copy()
    if poisoned.shape != (len(images.images),) or not poisoned.any():
        raise ValueError('Blend ASR requires recorded poisoned rows.')
    if not np.all(types[poisoned] == 'blended_injection') or not np.all(images.labels[poisoned] == target):
        raise ValueError('Blend type or target does not match saved recipe.')
    for index in np.flatnonzero(poisoned):
        expected = apply_image_trigger(clean_reference[int(index)][0].float(), spec)
        if not np.array_equal(expected.numpy(), images.images[index]):
            raise ValueError('Saved pixels do not match the blend recipe; ASR stopped.')
    return spec
