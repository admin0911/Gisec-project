"""Standalone feature extraction module for Gisec PoisonGuard."""

from .bundle import FeatureBundle
from .datasets import load_image_dataset, load_imdb_dataset, load_image_folder, load_text_table
from .detector import DetectorInput, detector_input
from .image_inputs import ImageInputBundle, load_image_inputs
from .packets import extract_packet_features
from .text import extract_text
from .universal import UniversalFeatureExtractor

__all__ = [
    "FeatureBundle", "UniversalFeatureExtractor", "load_image_dataset",
    "load_imdb_dataset", "load_image_folder", "load_text_table",
    "extract_text", "extract_packet_features",
    "DetectorInput", "detector_input",
    "ImageInputBundle", "load_image_inputs",
]
