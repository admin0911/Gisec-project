"""Serve the feature-layer frontend and a small local extraction API."""

import json
import os
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from poison_features import UniversalFeatureExtractor, load_image_dataset, load_imdb_dataset, extract_text
from poison_features.attacks import poison_dataset

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


class FeatureHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/datasets":
            self._json({
                "datasets": ["cifar10", "mnist", "imdb"],
                "attacks": ["none", "label_flip", "backdoor"],
                "text_attacks": ["none"],
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


def run_extraction(job_id: str, request: dict) -> None:
    try:
        name = request.get("dataset", "cifar10")
        limit = int(request.get("limit", 100))
        if limit < 2:
            raise ValueError("limit must be at least 2")
        attack = request.get("attack", "none")
        update_job(job_id, status="running", progress=2, message="Loading dataset")
        if name == "imdb":
            if attack != "none":
                raise ValueError("IMDB currently supports clean extraction only; text attacks are not implemented yet")
            update_job(job_id, status="running", progress=2, message="Loading IMDB reviews")
            data = load_imdb_dataset(split=request.get("split", "train"))
            total = min(limit, len(data))
            texts = [data[i]["text"] for i in range(total)]
            labels = [data[i]["label"] for i in range(total)]
            update_job(job_id, progress=15, message=f"Encoding {total:,} reviews with MiniLM")
            bundle = extract_text(
                texts, labels=labels, sample_ids=range(total), dataset_name=name,
            )
            update_job(job_id, progress=95, message="Preparing PCA visualization")
            result = {
                "dataset": name, "samples": len(bundle.features),
                "feature_dim": bundle.original_feature_dim,
                "reduced_dim": bundle.reduced_feature_dim,
                "visual_features": bundle.visual_features.tolist(),
                "labels": bundle.labels.tolist(),
                "poisoned": None,
                "poison_type": None,
            }
            update_job(job_id, status="complete", progress=100, message="Extraction complete", result=result)
            return
        dataset = load_image_dataset(name, train=request.get("split", "train") == "train")
        if attack != "none":
            dataset = poison_dataset(
                dataset, attack,
                poison_rate=float(request.get("poison_rate", 0.05)),
                seed=int(request.get("seed", 0)),
            )
        if limit < len(dataset):
            dataset = dataset.take(limit) if attack != "none" else __import__("torch").utils.data.Subset(dataset, range(limit))
        labels = [dataset[i][1] for i in range(len(dataset))]
        total = len(dataset)
        update_job(job_id, progress=5, message=f"Extracting {total:,} samples with ResNet-18")

        def progress(done, count):
            update_job(job_id, progress=5 + int(done / count * 85), message=f"Encoded {done:,} of {count:,} samples")

        bundle = UniversalFeatureExtractor(batch_size=32).extract_images(
            dataset, labels=labels, sample_ids=range(total), dataset_name=name, progress=progress,
        )
        update_job(job_id, progress=95, message="Preparing PCA visualization")
        result = {
            "dataset": name, "samples": len(bundle.features),
            "feature_dim": bundle.original_feature_dim,
            "reduced_dim": bundle.reduced_feature_dim,
            "visual_features": bundle.visual_features.tolist(),
            "labels": bundle.labels.tolist(),
            "poisoned": None if bundle.is_poisoned is None else int(bundle.is_poisoned.sum()),
            "poison_type": None if bundle.poison_type is None else bundle.poison_type.tolist(),
        }
        update_job(job_id, status="complete", progress=100, message="Extraction complete", result=result)
    except Exception as exc:
        update_job(job_id, status="error", progress=100, message=str(exc))


if __name__ == "__main__":
    frontend = Path(__file__).parent / "frontend"
    handler = partial(FeatureHandler, directory=str(frontend))
    port = int(os.environ.get("GISEC_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Gisec frontend: http://127.0.0.1:{port}")
    server.serve_forever()
