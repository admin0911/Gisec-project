"""Run lightweight end-to-end checks against the local real datasets."""

from poison_features import UniversalFeatureExtractor, load_image_dataset


def verify_image_dataset(name: str, count: int = 8) -> None:
    dataset = load_image_dataset(name, train=True, download=False)
    subset = __import__("torch").utils.data.Subset(dataset, range(count))
    labels = [dataset[i][1] for i in range(count)]
    bundle = UniversalFeatureExtractor(batch_size=4).extract_images(
        subset, labels=labels, sample_ids=range(count), dataset_name=name,
    )
    assert bundle.features.shape == (count, 512)
    assert bundle.scaled_features.shape == (count, 512)
    assert bundle.visual_features.shape == (count, 2)
    assert len(bundle.labels) == len(bundle.sample_ids) == count
    print(f"{name}: PASS; raw={bundle.features.shape}, PCA={bundle.reduced_features.shape}, visual={bundle.visual_features.shape}")


if __name__ == "__main__":
    verify_image_dataset("cifar10")
    verify_image_dataset("mnist")
