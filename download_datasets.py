"""Download the real image datasets used by the first feature milestones."""

from poison_features.datasets import load_image_dataset


if __name__ == "__main__":
    for dataset_name in ("cifar10", "mnist"):
        for train in (True, False):
            dataset = load_image_dataset(dataset_name, train=train)
            split = "train" if train else "test"
            print(f"{dataset_name} {split}: {len(dataset):,} samples")

