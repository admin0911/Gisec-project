"""Frozen CIFAR-10 raw-embedding cutoffs used in the 50,000-row benchmark."""
from copy import deepcopy

_CUTS = {
    'resnet18': {'knn': 1.0, 'class_distance': 0.024618882985227353,
                 'confident_learning': 0.9513872983583611},
    'dinov2': {'knn': 0.8, 'class_distance': 0.04322571427866734,
               'confident_learning': 0.7984671745392444},
}


def calibrated_profile(encoder):
    if encoder == 'dinov2_vits14':
        encoder = 'dinov2'
    if encoder not in _CUTS:
        raise ValueError('Choose resnet18 or dinov2 for the calibrated CIFAR-10 profile')
    return {'profile_name': 'cifar10_frozen_clean25k', 'version': '1.0',
            'encoder': encoder, 'feature_dimension': 512 if encoder == 'resnet18' else 384,
            'representation': 'raw_embeddings', 'thresholds': deepcopy(_CUTS[encoder]),
            'comparison': 'strict_greater', 'minimum_votes': 2, 'knn_k': 20,
            'confident_learning_requires_individual_flag': False,
            'confident_learning': {'folds': 5, 'seed': 2026, 'max_iter': 1000},
            'calibration_samples': 25000, 'split_seed': 20260910,
            'dependencies': {'numpy': '1.26.4', 'scikit-learn': '1.7.2', 'cleanlab': '2.7.1'},
            'status': 'development_calibration_full50k_includes_calibration_rows',
            'inactive_detectors': ['knn'] if encoder == 'resnet18' else []}
