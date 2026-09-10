"""Serve the feature-layer frontend and a small local extraction API."""

import json
import os
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from poison_features import (
    FeatureBundle,
    ImageInputBundle,
    UniversalFeatureExtractor,
    load_image_dataset,
    load_image_inputs,
    load_imdb_dataset,
    extract_text,
)
from poison_features.attacks import poison_dataset, poison_texts

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


class FeatureHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/datasets":
            self._json({
                "datasets": ["cifar10", "mnist", "imdb"],
                "attacks": ["none", "label_flip", "targeted_label_flip", "backdoor", "blended_injection"],
                "text_attacks": ["none", "label_flip", "targeted_label_flip", "backdoor"],
            })
            return
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self.send_error(404, "Unknown extraction job")
                return
            self._json(job)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/extract":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        name = request.get("dataset", "cifar10")
        limit = int(request.get("limit", 100))
        if limit < 2:
            raise ValueError("limit must be at least 2")
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0, "message": "Queued"}
        threading.Thread(
            target=run_extraction,
            args=(job_id, request),
            daemon=True,
        ).start()
        self._json({"job_id": job_id, "status": "queued"}, status=202)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(changes)


def bundle_result(bundle: FeatureBundle, feature_path: Path, image_path: Path | None) -> dict:
    return {
        "dataset": bundle.dataset_name,
        "samples": len(bundle.features),
        "feature_dim": bundle.original_feature_dim,
        "reduced_dim": bundle.reduced_feature_dim,
        "visual_features": bundle.visual_features.tolist()
        if bundle.visual_features is not None else [],
        "labels": bundle.labels.tolist(),
        "poisoned": None if bundle.is_poisoned is None else int(bundle.is_poisoned.sum()),
        "poison_type": None if bundle.poison_type is None else bundle.poison_type.tolist(),
        "feature_file": str(feature_path),
        "image_file": None if image_path is None else str(image_path),
    }


def run_extraction(job_id: str, request: dict) -> None:
    try:
        name = request.get("dataset", "cifar10")
        full_training = bool(request.get("full_training", False))
        limit = int(request.get("limit", 100))
        if full_training:
            limit = None
        if limit is not None and limit < 2:
            raise ValueError("limit must be at least 2")
        attack = request.get("attack", "none")
        encoder = request.get("encoder", "resnet18")
        if encoder not in {"resnet18", "dinov2"}:
            raise ValueError("encoder must be 'resnet18' or 'dinov2'")
        poison_rate = float(request.get("poison_rate", 0.05))
        source_label = request.get("source_label")
        source_label = None if source_label is None else int(source_label)
        target_label = int(request.get("target_label", 0))
        blend_alpha = float(request.get("blend_alpha", 0.10))
        poison_count = request.get("poison_count")
        poison_count = None if poison_count is None else int(poison_count)
        if attack == "none":
            poison_rate = 0.0
        if attack != "none" and poison_rate not in {0.01, 0.03, 0.05, 0.10}:
            raise ValueError("poison_rate must be 1%, 3%, 5%, or 10%")
        split = request.get("split", "train")
        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)
        size_key = "full" if full_training else str(limit)
        rate_key = f"{poison_rate:.2f}".replace(".", "")
        attack_key = (
            f"{attack}-s{source_label if source_label is not None else 'na'}"
            f"-a{blend_alpha:.2f}-t{target_label}-n{poison_count or 'rate'}"
        )
        update_job(job_id, status="running", progress=2, message="Loading dataset")
        if name == "imdb":
            stem = f"{name}-{split}-{size_key}-minilm-{attack_key}-{rate_key}-seed{int(request.get('seed', 0))}"
            feature_path = artifacts / f"{stem}-features.npz"
            if feature_path.exists():
                bundle = FeatureBundle.load(feature_path)
                update_job(
                    job_id,
                    status="complete",
                    progress=100,
                    message="Loaded saved extraction",
                    result=bundle_result(bundle, feature_path, None),
                )
                return
            update_job(job_id, status="running", progress=2, message="Loading IMDB reviews")
            data = load_imdb_dataset(split=split)
            rng = __import__("numpy").random.default_rng(int(request.get("seed", 0)))
            total = len(data) if limit is None else min(limit, len(data))
            indices = rng.choice(len(data), size=total, replace=False)
            texts = [data[int(i)]["text"] for i in indices]
            labels = [data[int(i)]["label"] for i in indices]
            metadata = {}
            if attack != "none":
                texts, labels, metadata = poison_texts(
                    texts, labels, attack=attack,
                    poison_rate=poison_rate,
                    source_label=source_label,
                    target_label=target_label,
                    seed=int(request.get("seed", 0)),
                )
            update_job(job_id, progress=15, message=f"Encoding {total:,} reviews with MiniLM")
            bundle = extract_text(
                texts, labels=labels, sample_ids=range(total), dataset_name=name,
                original_labels=metadata.get("original_labels"),
                is_poisoned=metadata.get("is_poisoned"),
                poison_type=metadata.get("poison_type"),
            )
            bundle.save(feature_path)
            update_job(job_id, progress=95, message="Preparing PCA visualization")
            result = bundle_result(bundle, feature_path, None)
            update_job(job_id, status="complete", progress=100, message="Extraction complete", result=result)
            return
        dataset = load_image_dataset(name, train=split == "train")
        clean_dataset = (
            __import__("torch").utils.data.Subset(dataset, range(limit))
            if limit is not None and limit < len(dataset)
            else dataset
        )
        attacked_dataset = clean_dataset
        if attack != "none":
            attacked_dataset = poison_dataset(
                dataset, attack,
                poison_rate=poison_rate,
                target_label=target_label,
                source_label=source_label,
                blend_alpha=blend_alpha,
                poison_count=poison_count,
                seed=int(request.get("seed", 0)),
            )
            if limit is not None and limit < len(attacked_dataset):
                attacked_dataset = attacked_dataset.take(limit)
        total = len(clean_dataset)
        sample_ids = np.asarray([f"{name}-{split}:{i}" for i in range(total)])
        labels = np.asarray([attacked_dataset[i][1] for i in range(total)])
        metadata = getattr(attacked_dataset, "metadata", None)
        encoders = ["resnet18", "dinov2"] if name == "cifar10" else [encoder]
        results = []
        clean_images_path = artifacts / f"{name}-{split}-{size_key}-images.npz"
        if not clean_images_path.exists():
            load_image_inputs(clean_dataset, sample_ids=sample_ids).save(clean_images_path)

        for encoder_name in encoders:
            encoder_key = encoder_name
            clean_stem = f"{name}-{split}-{size_key}-{encoder_key}-none-a{0.0:.2f}-t0-nrate-000-seed{int(request.get('seed', 0))}"
            clean_feature_path = artifacts / f"{clean_stem}-features.npz"
            attack_stem = f"{name}-{split}-{size_key}-{encoder_key}-{attack_key}-seed{int(request.get('seed', 0))}"
            feature_path = artifacts / f"{attack_stem}-features.npz"
            image_path = artifacts / f"{attack_stem}-images.npz"
            if attack in {"label_flip", "targeted_label_flip"} and clean_feature_path.exists():
                base = FeatureBundle.load(clean_feature_path)
                bundle = FeatureBundle(
                    features=base.features,
                    scaled_features=base.scaled_features,
                    reduced_features=base.reduced_features,
                    labels=labels,
                    sample_ids=sample_ids,
                    modality=base.modality,
                    encoder=base.encoder,
                    dataset_name=base.dataset_name,
                    visual_features=base.visual_features,
                    original_labels=None if metadata is None else metadata.original_labels[:total],
                    is_poisoned=None if metadata is None else metadata.is_poisoned[:total],
                    poison_type=None if metadata is None else metadata.poison_type[:total],
                    metadata={**(base.metadata or {}), "attack": attack, "reused_clean_features": True},
                )
                bundle.save(feature_path)
                if not image_path.exists():
                    load_image_inputs(clean_dataset, sample_ids=sample_ids).save(image_path)
            elif feature_path.exists():
                bundle = FeatureBundle.load(feature_path)
            else:
                source = clean_dataset if attack == "none" else attacked_dataset
                update_job(job_id, progress=5, message=f"Extracting {encoder_name} features")

                def progress(done, count):
                    update_job(job_id, progress=5 + int(done / count * 85), message=f"{encoder_name}: encoded {done:,} of {count:,}")

                bundle = UniversalFeatureExtractor(batch_size=32).extract_images(
                    source, labels=labels, sample_ids=sample_ids, dataset_name=name,
                    encoder=encoder_name, progress=progress,
                )
                bundle.metadata.update({
                    "encoder": encoder_name,
                    "attack": attack,
                    "poison_rate": poison_rate,
                    "target_label": target_label,
                    "blend_alpha": blend_alpha,
                    "poison_count": int(bundle.is_poisoned.sum()) if bundle.is_poisoned is not None else 0,
                })
                bundle.save(feature_path)
                if not image_path.exists():
                    load_image_inputs(source, sample_ids=sample_ids).save(image_path)
            results.append(bundle_result(bundle, feature_path, image_path))
        update_job(job_id, progress=95, message="Preparing PCA visualization")
        primary = results[0]
        primary["representations"] = results
        update_job(job_id, status="complete", progress=100, message="Extraction complete", result=primary)
    except Exception as exc:
        update_job(job_id, status="error", progress=100, message=str(exc))


if __name__ == "__main__":
    frontend = Path(__file__).parent / "frontend"
    handler = partial(FeatureHandler, directory=str(frontend))
    port = int(os.environ.get("GISEC_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Gisec frontend: http://127.0.0.1:{port}")
    server.serve_forever()
