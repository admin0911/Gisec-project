"""New, swappable image encoder implementation."""

from pathlib import Path
from typing import Any, Callable

import numpy as np


class ResNet18ImageEncoder:
    """Extract frozen 512-dimensional ImageNet ResNet-18 embeddings."""

    def __init__(self, device: str | None = None, weights_dir: str | Path = "artifacts/weights"):
        try:
            import torch
            from torch import nn
            from torchvision.models import ResNet18_Weights, resnet18
            from torchvision.transforms import InterpolationMode, Resize, Normalize
        except ImportError as exc:
            raise RuntimeError(
                "Image extraction requires torch and torchvision; install requirements.txt"
            ) from exc

        self.torch = torch
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights = ResNet18_Weights.DEFAULT
        with torch.random.fork_rng(devices=[]):
            model = resnet18(weights=weights)
        model.fc = nn.Identity()
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.transform = lambda image: Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )(Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(image))
        self._to_tensor: Callable[[Any], Any] = (
            __import__("torchvision").transforms.functional.to_tensor
        )

    def _prepare(self, image: Any) -> Any:
        from PIL import Image

        if isinstance(image, Image.Image):
            tensor = self._to_tensor(image.convert("RGB"))
        elif isinstance(image, self.torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
                raise ValueError("Expected a CHW tensor with 1 or 3 channels")
            if tensor.dtype == self.torch.uint8:
                tensor = tensor.float() / 255
            elif tensor.is_floating_point():
                tensor = tensor.float()
            else:
                raise ValueError("Tensor pixels must be uint8 or floating-point")
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
        else:
            raise TypeError("Images must be PIL images or CHW tensors")
        if tensor.numel() == 0 or not self.torch.isfinite(tensor).all():
            raise ValueError("Images must contain finite, non-empty pixels")
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Floating-point image pixels must be in [0, 1]")
        return self.transform(tensor)

    def extract(self, dataset: Any, batch_size: int = 128, progress: Any = None) -> np.ndarray:
        from torch.utils.data import DataLoader, Dataset

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        encoder = self

        class Images(Dataset):
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, index: int) -> Any:
                item = dataset[index]
                return encoder._prepare(item[0] if isinstance(item, (tuple, list)) else item)

        loader = DataLoader(Images(), batch_size=batch_size, shuffle=False, num_workers=0)
        chunks = []
        with self.torch.inference_mode():
            for images in loader:
                chunks.append(self.model(images.to(self.device)).cpu().numpy())
                if progress:
                    progress(sum(len(chunk) for chunk in chunks), len(dataset))
        result = np.concatenate(chunks).astype(np.float32)
        if result.shape[1] != 512 or not np.isfinite(result).all():
            raise RuntimeError("ResNet-18 returned invalid features")
        return result


class DINOv2ImageEncoder(ResNet18ImageEncoder):
    """384-dimensional CLS embeddings, with pinned official source code.

    First use downloads official code and weights through torch.hub.
    Full-image 224x224 resizing preserves the field of view (no centre crop).
    """
    revision = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"

    def __init__(self, device=None, weights_dir="artifacts/weights"):
        import torch
        from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
        from torchvision.transforms.functional import to_tensor

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        # Do not change torch.hub's process-wide cache configuration here.
        with torch.random.fork_rng(devices=[]):
            model = torch.hub.load(
                f"facebookresearch/dinov2:{self.revision}", "dinov2_vits14",
                trust_repo=True, pretrained=True,
            )
        self.model = model.to(self.device).eval().requires_grad_(False)
        self._to_tensor = to_tensor
        self.transform = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def extract(self, dataset, batch_size=32, progress=None):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not len(dataset):
            raise ValueError("Image dataset must not be empty")
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                images = []
                for index in range(start, min(start + batch_size, len(dataset))):
                    item = dataset[index]
                    images.append(self._prepare(item[0] if isinstance(item, (tuple, list)) else item))
                # Run each sample independently so reduction kernels cannot
                # change the embedding by batch size or execution shape.
                values = self.torch.cat([
                    self.model(image.unsqueeze(0).to(self.device))
                    for image in images
                ]).cpu().numpy()
                if values.shape != (len(images), 384) or not np.isfinite(values).all():
                    raise RuntimeError("DINOv2 returned invalid features")
                chunks.append(values.astype(np.float32))
                if progress:
                    progress(min(start + batch_size, len(dataset)), len(dataset))
        return np.concatenate(chunks)
