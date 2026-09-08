"""Serve the feature-layer frontend and a small local extraction API."""

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from poison_features import UniversalFeatureExtractor, load_image_dataset
from poison_features.attacks import poison_dataset


class FeatureHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/datasets":
            self._json({"datasets": ["cifar10", "mnist"], "attacks": ["none", "label_flip", "backdoor"]})
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
        dataset = load_image_dataset(name, train=request.get("split", "train") == "train")
        attack = request.get("attack", "none")
        if attack != "none":
            dataset = poison_dataset(
                dataset,
                attack,
                poison_rate=float(request.get("poison_rate", 0.05)),
                seed=int(request.get("seed", 0)),
            )
        if limit < len(dataset):
            dataset = dataset.take(limit) if attack != "none" else __import__("torch").utils.data.Subset(dataset, range(limit))
        labels = [dataset[i][1] for i in range(len(dataset))]
        bundle = UniversalFeatureExtractor(batch_size=32).extract_images(
            dataset, labels=labels, sample_ids=range(len(dataset)), dataset_name=name,
        )
        self._json({
            "dataset": name,
            "samples": len(bundle.features),
            "feature_dim": bundle.original_feature_dim,
            "reduced_dim": bundle.reduced_feature_dim,
            "visual_features": bundle.visual_features.tolist(),
            "labels": bundle.labels.tolist(),
            "poisoned": None if bundle.is_poisoned is None else int(bundle.is_poisoned.sum()),
            "poison_type": None if bundle.poison_type is None else bundle.poison_type.tolist(),
        })

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    frontend = Path(__file__).parent / "frontend"
    handler = partial(FeatureHandler, directory=str(frontend))
    port = int(os.environ.get("GISEC_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Gisec frontend: http://127.0.0.1:{port}")
    server.serve_forever()
