"""Small from-scratch image baseline for wiring and training comparisons."""
from torch import nn


def small_cnn(channels=3, num_classes=10):
    return nn.Sequential(
        nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes),
    )
