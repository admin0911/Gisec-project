import unittest

import numpy as np

from poison_features.attacks import poison_dataset, poison_texts
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

    def test_blended_injection_uses_shared_noise_and_target_label(self):
        import torch

        dataset = [(torch.full((3, 8, 8), 0.2), label) for label in [1, 2, 3, 4]]
        poisoned = poison_dataset(
            dataset,
            "blended_injection",
            poison_count=2,
            target_label=0,
            blend_alpha=0.10,
            seed=4,
        )
        changed = [
            poisoned[index][0] - dataset[index][0]
            for index in range(len(dataset))
            if poisoned.metadata.is_poisoned[index]
        ]
        self.assertEqual(int(poisoned.metadata.is_poisoned.sum()), 2)
        self.assertTrue(np.all(poisoned.metadata.current_labels[poisoned.metadata.is_poisoned] == 0))
        self.assertTrue(torch.allclose(changed[0], changed[1]))
        self.assertLess(float(changed[0].abs().max()), 0.1)

    def test_imdb_label_flip_preserves_text_and_separates_metadata(self):
        texts = ["bad review", "good review", "mixed review", "another review"]
        labels = np.array([0, 1, 0, 1])
        changed_texts, changed_labels, metadata = poison_texts(
            texts, labels, attack="label_flip", poison_rate=0.5, seed=3,
        )
        self.assertEqual(changed_texts, texts)
        self.assertEqual(int(metadata["is_poisoned"].sum()), 2)
        np.testing.assert_array_equal(
            changed_labels[metadata["is_poisoned"]],
            1 - labels[metadata["is_poisoned"]],
        )
        np.testing.assert_array_equal(metadata["original_labels"], labels)
        self.assertTrue(np.all(metadata["poison_type"][~metadata["is_poisoned"]] == "clean"))

    def test_imdb_backdoor_adds_trigger_and_target_label(self):
        texts = ["bad review", "good review", "mixed review", "another review"]
        labels = np.array([0, 1, 0, 1])
        trigger = "test trigger"
        changed_texts, changed_labels, metadata = poison_texts(
            texts,
            labels,
            attack="backdoor",
            poison_rate=0.5,
            target_label=1,
            trigger=trigger,
            seed=3,
        )
        poisoned = metadata["is_poisoned"]
        self.assertEqual(int(poisoned.sum()), 2)
        self.assertTrue(all(trigger in changed_texts[index] for index in np.flatnonzero(poisoned)))
        self.assertTrue(all(changed_texts[index] == texts[index] for index in np.flatnonzero(~poisoned)))
        np.testing.assert_array_equal(changed_labels[poisoned], np.ones(2, dtype=np.int64))
        self.assertTrue(np.all(metadata["poison_type"][poisoned] == "backdoor"))


if __name__ == "__main__":
    unittest.main()
