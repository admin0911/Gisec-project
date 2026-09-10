"""Confident Learning with five-fold out-of-fold logistic predictions.
Dependency: cleanlab (see requirements.txt).
Frozen raw embeddings are normalized per image (no cross-fold fitting).
Labels train diagnostic classifiers; poison identities never enter this class.
"""
import warnings
import numpy as np


class ConfidentLearning:
    def __init__(self, folds=5, seed=2026, max_iter=1000):
        if not isinstance(folds, int) or folds < 3:
            raise ValueError("Use at least three folds")
        self.folds, self.seed, self.max_iter = folds, seed, max_iter

    def analyze(self, inputs, progress=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.exceptions import ConvergenceWarning
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
        from threadpoolctl import threadpool_limits
        X = np.asarray(inputs.X, dtype=np.float64)
        labels = np.asarray(inputs.y)
        ids = np.asarray(inputs.sample_ids)
        if X.ndim != 2 or not X.size or not np.isfinite(X).all():
            raise ValueError("Features must be finite and two-dimensional")
        n = len(X)
        if inputs.y is None or labels.shape != (n,) or labels.dtype.kind not in "biufUS":
            raise ValueError("Known current labels are required")
        if labels.dtype.kind in "biuf" and (np.any(labels == -1) or not np.isfinite(labels).all()):
            raise ValueError("Labels must be known and finite")
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("IDs must be unique and aligned")
        classes, y, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(classes) < 2 or np.min(counts) < self.folds:
            raise ValueError("Each of at least two classes needs at least folds samples")
        scale = np.max(np.abs(X), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("Zero feature vectors are unsupported")
        X = X / scale
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        probabilities = np.empty((n, len(classes)))
        fold_ids = np.full(n, -1, dtype=int)
        iterations = []
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        with threadpool_limits(limits=2):
            for fold, (train, test) in enumerate(splitter.split(X, y)):
                if progress:
                    progress(f"Diagnostic fold {fold + 1}/{self.folds}")
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=self.max_iter, tol=1e-4)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(X[train], y[train])
                np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))
                assert not np.intersect1d(train, test).size
                assert np.all(fold_ids[test] == -1)
                probabilities[test] = model.predict_proba(X[test])
                fold_ids[test] = fold
                iterations.append(int(np.max(model.n_iter_)))
        assert np.all(fold_ids >= 0)
        flags = find_label_issues(y, probabilities, filter_by="prune_by_noise_rate", n_jobs=1)
        quality = get_label_quality_scores(y, probabilities, method="self_confidence")
        return dict(sample_ids=ids.copy(), scores=1 - quality, flags=flags,
                    pred_probs=probabilities, classes=classes, fold_ids=fold_ids,
                    alternative_labels=classes[np.argmax(probabilities, axis=1)],
                    iterations=np.asarray(iterations))

    def score(self, inputs):
        return self.analyze(inputs)["scores"]
