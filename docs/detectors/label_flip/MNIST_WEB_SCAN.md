# MNIST scanning in the web app

Choose MNIST, scope and attack, then Build dataset. The web app temporarily
saves only the image bundle and skips encoder extraction and PCA for MNIST.
Set `MNIST_PIXELS_ONLY = false` in `frontend/app.js` to restore extraction later.
The shared backend extractor remains available when `pixels_only` is omitted
or false. CIFAR extraction is unchanged. Known poison metadata is saved separately.

Scan label flips resolves the matching `*-images.npz` through ImageInputBundle.
It reads original 28x28 pixels, current labels and stable training sample IDs.
The feature artifact's public IDs and labels are checked for alignment; its
embedding vectors and poison metadata are not used for scoring.

The three existing detectors use the same pixel settings as the standalone
experiment: cosine kNN 19/20, class distance 0.1, and five-fold out-of-fold
Confident Learning. These settings are provisional, not MNIST-calibrated.
No detector is selected based on the known attack setting.

Results use zero votes for not flagged, one vote for needs review, and at least
two votes for suspected label flipping. These are review assessments, not proof
of clean or poisoned data. The three individual outputs use the shared detector
dictionary format. The original results and their file hashes are saved; matching
completed scans are reused. Human review shows digit names and pixel vote counts.

CIFAR-10 continues to use its two-encoder scanning path. Training comparison is
still CIFAR-10-only, so the training link is hidden on MNIST results.

Validation: unit tests cover connector alignment, truth isolation, vote mapping,
restoration and human review. A 200-image live build/scan checks integration only;
it is not a replacement for the separate 60,000-image MNIST benchmark.
