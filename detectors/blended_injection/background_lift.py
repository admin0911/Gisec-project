"""Detect blended injection on datasets with a constant dark background.

Some image collections have pixels that are reliably zero throughout - MNIST's
border is pure black in essentially every digit. Blending noise into an image
cannot leave those pixels at zero, so a poisoned row lights up pixels that are
dark in every clean row.

This is a different tell from the shared-pattern signature in
`residual_signature`, and it works precisely where that one fails. The shared
pattern approach needs images to be individually distinctive, which holds for
photographs and not for handwritten digits, where every example of a class
already resembles the others. This check does not care about resemblance at
all; it asks only whether pixels that should be empty are empty.

It applies only where such pixels exist. Photographs rarely contain exact
zeros, so on CIFAR-10 the background is empty and the check reports nothing
rather than inventing a signal.
"""

import numpy as np

# A pixel counts as background when it is zero in at least this share of
# images. Poison rates reach 10%, so the bar allows for the poisoned rows
# themselves having lifted it without disqualifying the pixel.
_DARK_FRACTION = 0.80

# Below this many background pixels the dataset has no usable dark region and
# the check does not apply.
_MIN_DARK_PIXELS = 32

# An image is suspect when this share of the background pixels are lit. Clean
# rows sit near zero and blended rows near one, so the midpoint is ample.
_LIT_FRACTION = 0.5

# A genuine attack marks many rows, and it targets one class rather than most
# of the dataset.
_MIN_GROUP = 3
_MAX_FLAGGED_FRACTION = 0.6

# The suspect class must carry this many times the flagged share of the next
# worst class. Blended injection relabels into a single class, so the others
# act as a control group.
_MIN_CLASS_RATIO = 5.0


def background_mask(images: np.ndarray, dark_fraction: float = _DARK_FRACTION) -> np.ndarray:
    """Return a boolean mask of pixels that are zero across most of the dataset."""
    flat = images.reshape(len(images), -1)
    return (flat == 0).mean(axis=0) >= dark_fraction


def lift_scores(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fraction of background pixels that are lit, per image."""
    flat = images.reshape(len(images), -1)
    if not mask.any():
        return np.zeros(len(images), dtype=np.float64)
    return (flat[:, mask] != 0).mean(axis=1).astype(np.float64)


def scan_background_lift(pixels, dark_fraction: float = _DARK_FRACTION,
                         lit_fraction: float = _LIT_FRACTION,
                         min_class_ratio: float = _MIN_CLASS_RATIO) -> dict:
    """Look for a class whose rows light up the dataset's dark background.

    Returns a dict shaped like the other scans, with `target_class` set to None
    when the dataset has no usable background or no class stands out.
    """
    images = pixels.images
    labels = np.asarray(pixels.labels)
    sample_ids = np.asarray(pixels.sample_ids)
    n = len(images)

    nothing = {
        "sample_ids": sample_ids,
        "scores": np.zeros(n, dtype=np.float64),
        "flags": np.zeros(n, dtype=bool),
        "target_class": None,
        "background_pixels": 0,
        "class_lit_share": {},
        "class_ratio": 0.0,
        "method": "background-lift",
    }

    mask = background_mask(images, dark_fraction)
    dark_pixels = int(mask.sum())
    if dark_pixels < _MIN_DARK_PIXELS:
        return nothing

    scores = lift_scores(images, mask)
    lit = scores > lit_fraction

    shares = {}
    for label in np.unique(labels):
        in_class = labels == label
        shares[int(label)] = float(lit[in_class].mean()) if in_class.any() else 0.0

    nothing["background_pixels"] = dark_pixels
    nothing["class_lit_share"] = shares
    if not shares:
        return nothing

    suspect = max(shares, key=shares.get)
    ordered = sorted(shares.values(), reverse=True)
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    ratio = (shares[suspect] / runner_up) if runner_up > 0 else float("inf")

    flags = np.zeros(n, dtype=bool)
    flags[(labels == suspect) & lit] = True
    group = int(flags.sum())

    if group < _MIN_GROUP or ratio < min_class_ratio:
        return nothing
    if flags.mean() > _MAX_FLAGGED_FRACTION:
        return nothing

    return {
        "sample_ids": sample_ids,
        "scores": np.where(labels == suspect, scores, 0.0),
        "flags": flags,
        "target_class": suspect,
        "group_size": group,
        "background_pixels": dark_pixels,
        "class_lit_share": shares,
        "class_ratio": ratio,
        "method": "background-lift",
    }
