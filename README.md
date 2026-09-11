# PoisonGuard

PoisonGuard scans training datasets for suspicious labels, repeated backdoor
triggers and blended noise, then compares training before and after filtering.
Human review is optional. A flag does not prove poisoning, and no flags do not
guarantee a clean dataset.

## Setup and launch (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_datasets.py
.\.venv\Scripts\python.exe serve_frontend.py
```

Open http://127.0.0.1:8787. Image datasets are downloaded into `data/`.
IMDB and pretrained encoders require downloads on first use. Keep the dependency
pins in `requirements.txt`; data, weights and generated artifacts remain local.

## Main workflow

1. **Build:** choose a dataset, Quick sample or Full training dataset, and an
   attack configuration. Saved matching extraction artifacts can be reused.
2. **Scan:** select the saved dataset and run its applicable checks. A matching
   completed scan may reopen from cache; extraction and scan caches are separate.
3. **Review and prepare:** optionally save human decisions, then prepare the
   training selection. Quarantined and unresolved samples are excluded without
   deleting the original data.
4. **Train:** compare clean reference, before cleaning and after cleaning.
   Use the three-seed benchmark for reported comparisons.

| Dataset | Scan inputs | Attack scenarios |
|---|---|---|
| CIFAR-10 | ResNet-18 and DINOv2 features; original-sized pixels | Label flip, targeted label flip, patch, blended noise, mixed attacks |
| MNIST | Pixels; web build skips image encoders | Label flip, targeted label flip, patch, blended noise, mixed attacks |
| IMDB | MiniLM features and review text | Label flip and repeated-phrase backdoor |

The UI's Random label flip randomly selects rows but assigns the next class;
it does not randomly choose replacement labels. Mixed image attacks use separate
sample groups with a total poisoning rate of at most 10%. Synthetic attack
identities are used for evaluation only, never to decide detector flags.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Get-ChildItem tests/test_*.cjs | ForEach-Object { node $_.FullName; if ($LASTEXITCODE -ne 0) { throw "Frontend test failed" } }
```

The frontend tests require Node.js. `python verify_extractor.py` additionally
checks extraction using locally downloaded image data. Small fixture tests do
not establish real-data detection performance.

## Documentation

- [Training](docs/TRAINING.md): selection, defaults, seeds, reports and ASR support.
- [Detectors](docs/DETECTORS.md): dataset routing and detailed method guides.
- [Evaluation](docs/EVALUATION.md): calibration, held-out evaluation and reporting.
- [Connectors](docs/CONNECTORS.md): integration contracts and sample identity.
- [Feature extraction](docs/FEATURES.md): custom datasets and Python examples.
- [Contributing](CONTRIBUTING.md): development and Git workflow.

Detailed experiment notes are linked from these guides. Generated results live
under `artifacts/`; they are not included in a fresh Git clone.
