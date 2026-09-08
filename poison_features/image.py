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

