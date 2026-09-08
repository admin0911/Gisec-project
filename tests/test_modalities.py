import unittest

import numpy as np

from poison_features.attacks import poison_dataset
from poison_features.packets import extract_packet_features


class _TinyImages:
    def __init__(self):
        import torch
        self.items = [(torch.zeros(1, 8, 8), 1) for _ in range(10)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ModalityTests(unittest.TestCase):
    def test_attack_metadata_is_separate(self):
        poisoned = poison_dataset(_TinyImages(), "label_flip", poison_rate=0.2, seed=4)
        self.assertEqual(int(poisoned.metadata.is_poisoned.sum()), 2)
        self.assertEqual(len(poisoned.metadata.current_labels), 10)
        self.assertTrue(np.all(poisoned.metadata.poison_type[~poisoned.metadata.is_poisoned] == "clean"))

    def test_packet_bundle_shape_and_finiteness(self):
        records = [
            {"duration": 1, "packet_count": 3, "byte_count": 1000, "protocol": "tcp"},
            {"duration": 2, "packet_count": 4, "byte_count": 1200, "protocol": "udp"},
            {"duration": 3, "packet_count": 5, "byte_count": 1400, "protocol": "tcp"},
        ]
        bundle = extract_packet_features(records)
        self.assertEqual(bundle.features.shape, (3, 12))
        self.assertEqual(bundle.reduced_features.shape, (3, 2))
        self.assertTrue(np.isfinite(bundle.scaled_features).all())


if __name__ == "__main__":
    unittest.main()
