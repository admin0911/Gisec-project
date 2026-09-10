# Detector guide

The scan runs the checks applicable to the dataset, rather than choosing a
detector from the injected attack's ground truth. Scores express suspicion,
not poisoning probabilities, and cannot be averaged across methods.

| Dataset | Label checks | Additional checks |
|---|---|---|
| CIFAR-10 | kNN, class distance and confident learning on each of ResNet-18 and DINOv2 | Repeated patch; blended-injection pipeline |
| MNIST | Three pixel-based label checks | Repeated patch; blended-injection pipeline |
| IMDB | Three MiniLM-based label checks | Repeated phrase |

Both image datasets use the shared blended pipeline, with residual-signature
analysis and a background-lift fallback. The old consensus-pixel detector has
been retired. Natural-image and constant-background methods have different
assumptions; a successful synthetic case does not establish general detection.

## Method and experiment details

- [Blended injection](../detectors/blended_injection/README.md)
- [Repeated patch](detectors/backdoor/REPEATED_PATCH.md)
- [Repeated phrase](detectors/backdoor/REPEATED_PHRASE.md)
- [CIFAR web scan](detectors/label_flip/WEB_SCAN.md)
- [IMDB](detectors/label_flip/IMDB.md)
- [MNIST web scan](detectors/label_flip/MNIST_WEB_SCAN.md) and [pixel experiments](detectors/label_flip/MNIST_PIXELS.md)
- [Confident learning](detectors/label_flip/CONFIDENT_LEARNING.md)
- [Feature comparison](detectors/label_flip/FEATURE_COMPARISON.md)
- [Assessment](detectors/label_flip/ASSESSMENT.md), [cleaning APIs](detectors/label_flip/CLEANING.md) and [human review](detectors/label_flip/HUMAN_REVIEW.md)

See [Connectors](CONNECTORS.md) for integration, [Training](TRAINING.md) for the
current selection workflow and [Evaluation](EVALUATION.md) before quoting metrics.
Detailed experiment presets describe their own scope, not universal thresholds.
