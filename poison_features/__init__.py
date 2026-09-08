"""Standalone feature extraction module for Gisec PoisonGuard."""

from .bundle import FeatureBundle
from .datasets import load_image_dataset
from .universal import UniversalFeatureExtractor

__all__ = ["FeatureBundle", "UniversalFeatureExtractor", "load_image_dataset"]
