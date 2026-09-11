# Training connector

`training/connector.py` accepts a dataset view, stable sample IDs, dataset version
and split. Each item is `(input_tensor, current_label)`. No detector scores,
thresholds or poison identities enter training. Use the actual post-attack images
and supplied labels. A teammate can consume TrainingInput in another trainer.

```python
from cleaning.label_flip import partition_dataset
from training import training_input, train_classifier, TrainConfig
from training.models import small_cnn

views = partition_dataset(dataset, sample_ids, decisions)
kept = views['keep']
train = training_input(kept, kept.sample_ids,
                       dataset_version='cifar10-run-001', split='train')
test = training_input(test_dataset, test_ids,
                      dataset_version='cifar10-official-test', split='test')
report = train_classifier(train, test,
    model_factory=lambda: small_cnn(channels=3, num_classes=10),
    model_name='small_cnn_v1', num_classes=10,
    config=TrainConfig(epochs=10, seed=42), output_root='artifacts/training',
    progress=print)
```

Variables dataset, IDs, decisions and test_dataset come from the caller. The
model factory must construct a new untrained torch module returning class logits.
For MNIST use channels=1. Inputs to the provided CNN are original-sized float32
pixels in [0,1], not ResNet/DINO embeddings. Small CNN is a baseline, not a claim
of optimal CIFAR-10 accuracy. Existing extractors stay frozen and separate.

The trainer uses fixed epochs, Adam and cross-entropy, then evaluates once on the
test split. Use validation data in separate experiments for tuning; do not choose
epochs or thresholds using test results. Seeds are recorded, but exact matching
across hardware is not guaranteed. ID overlap is rejected; the caller must also
ensure IDs are honest, split-qualified and correspond to distinct source data.

## CLI using saved files

```powershell
python -m training.run --images artifacts/YOUR-images.npz --decisions artifacts/YOUR-RUN/decisions.json --dataset cifar10 --dataset-version YOUR-RUN --epochs 10
```

Replace placeholders with matching artifacts from the same dataset version.
The manifest must cover every image ID. Only keep rows enter training; all other
rows remain saved outside the training view. The official test dataset must
already be downloaded under data/. Use --device cuda if available.

## Standard training result

The returned dictionary has schema_version, trainer_name, version, model_name,
num_classes, settings, dataset_versions, training_samples, test_samples, history,
metrics, seconds, torch_version and artifacts. Metrics include accuracy (0 to 1),
per_class_accuracy, class_support and a confusion matrix (true rows, predicted
columns). Missing classes receive null accuracy. No poison detection metrics are
claimed by this trainer.

Each run saves report.json, model.pt (state dictionary), training_ids.npy and
test_predictions.npz with sample_ids, predictions and labels. Reconstruct the
recorded model architecture before loading weights. Never load untrusted files.

To compare filtering policies, construct different dataset views externally and
call the same trainer with matching model/settings/seeds/test split. A clean
reference run belongs to benchmark code, not to the real cleaning decision.
Backdoor ASR evaluation and a full policy comparison are later steps; this module
currently reports classification on the provided test dataset only.

The controlled square-patch defence supplies those additional steps through
`python -m experiments.defend_backdoor`; see
[`detectors/backdoor/DEFENCE.md`](detectors/backdoor/DEFENCE.md).

The frontend is not wired to this entry point yet. No automatic git push occurs.
