"""Fresh classifier training; evaluation runs once after the fixed epoch budget."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .connector import training_input


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 42
    device: str = 'cpu'


def _batch(batch, device, num_classes):
    x, y = batch
    if not torch.is_tensor(x) or not torch.isfinite(x).all():
        raise ValueError('Model inputs must be finite tensors')
    if y.ndim != 1 or y.dtype not in (torch.int32, torch.int64):
        raise ValueError('Labels must be integer class indices')
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError('Label outside configured classes')
    return x.to(device), y.to(device=device, dtype=torch.long)


def train_classifier(train, test, *, model_factory, model_name, num_classes,
                     output_root, config=None, progress=None):
    """Factory must return a NEW untrained torch module on every call.

    Teammates may replace model_factory. TrainingInput and the returned summary
    form the shared interface. No detector or poison metadata is consulted.
    """
    config = config or TrainConfig()
    for value in (train, test):
        training_input(value.dataset, value.sample_ids,
                       dataset_version=value.dataset_version, split=value.split)
    if train.split != 'train' or test.split != 'test':
        raise ValueError('Provide a training split and a separate test split')
    if set(train.sample_ids.tolist()) & set(test.sample_ids.tolist()):
        raise ValueError('Training and test sample IDs overlap')
    if (type(config.epochs) is not int or config.epochs < 1
            or type(config.batch_size) is not int or config.batch_size < 1
            or not np.isfinite(config.learning_rate) or config.learning_rate <= 0
            or type(config.seed) is not int or config.seed < 0
            or type(num_classes) is not int or num_classes < 2 or not model_name):
        raise ValueError('Invalid training configuration')
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train.dataset, batch_size=config.batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test.dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)
    history = []
    start = time.monotonic()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for batch in loader:
            x, y = _batch(batch, device, num_classes)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.shape != (len(y), num_classes) or not torch.isfinite(logits).all():
                raise ValueError('Model must return finite N by num_classes logits')
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        row = {'epoch': epoch + 1, 'loss': total_loss / count,
               'accuracy': correct / count}
        history.append(row)
        if progress:
            progress(f'Epoch {epoch + 1}/{config.epochs}: loss {row["loss"]:.4f}')
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in test_loader:
            x, y = _batch(batch, device, num_classes)
            logits = model(x)
            if logits.shape != (len(y), num_classes) or not torch.isfinite(logits).all():
                raise ValueError('Invalid evaluation logits')
            predictions.append(logits.argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (target, pred), 1)
    supports = confusion.sum(1)
    metrics = {'accuracy': float(np.mean(pred == target)),
               'per_class_accuracy': [float(confusion[i, i] / n) if n else None
                                      for i, n in enumerate(supports)],
               'class_support': supports.tolist(),
               'confusion_matrix': confusion.tolist()}
    out = Path(output_root) / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'schema_version': '1.0', 'trainer_name': 'torch_classifier', 'version': '1.0',
        'model_name': model_name, 'num_classes': num_classes, 'settings': asdict(config),
        'training_samples': len(train.dataset), 'test_samples': len(test.dataset),
        'dataset_versions': {'train': train.dataset_version, 'test': test.dataset_version},
        'history': history, 'metrics': metrics, 'seconds': time.monotonic() - start,
        'torch_version': str(torch.__version__),
        'artifacts': {name: str((out / filename).resolve()) for name, filename in {
            'report': 'report.json', 'checkpoint': 'model.pt',
            'predictions': 'test_predictions.npz', 'training_ids': 'training_ids.npy'}.items()},
    }
    torch.save(model.cpu().state_dict(), out / 'model.pt')
    np.save(out / 'training_ids.npy', train.sample_ids, allow_pickle=False)
    np.savez_compressed(out / 'test_predictions.npz', sample_ids=test.sample_ids,
                        predictions=pred, labels=target)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
