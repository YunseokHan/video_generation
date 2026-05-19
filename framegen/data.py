from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

try:
    import numpy as np
except ImportError:  # pragma: no cover - torch environments generally include numpy.
    np = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_frame_positions(num_frames: int, normalize: bool = True) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not normalize:
        return torch.arange(num_frames, dtype=torch.float32)
    if num_frames == 1:
        return torch.zeros(1, dtype=torch.float32)
    return torch.linspace(0.0, 1.0, steps=num_frames, dtype=torch.float32)


def flatten_video_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    frames = batch["frames"]
    captions = batch["captions"]
    frame_positions = batch["frame_positions"]

    if frames.ndim != 5:
        raise ValueError(f"Expected frames [B, F, 3, H, W], got {tuple(frames.shape)}")
    if frame_positions.ndim != 2:
        raise ValueError(f"Expected frame_positions [B, F], got {tuple(frame_positions.shape)}")

    batch_size, num_frames = frames.shape[:2]
    frames_flat = frames.reshape(batch_size * num_frames, *frames.shape[2:])
    captions_flat = [caption for caption in captions for _ in range(num_frames)]
    frame_positions_flat = frame_positions.reshape(batch_size * num_frames)

    assert len(captions_flat) == frames_flat.shape[0]
    assert frame_positions_flat.shape[0] == frames_flat.shape[0]

    return frames_flat, captions_flat, frame_positions_flat


def video_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": torch.stack([item["frames"] for item in items], dim=0),
        "captions": [item["caption"] for item in items],
        "frame_positions": torch.stack([item["frame_positions"] for item in items], dim=0),
    }


class PlaceholderVideoDataset(Dataset):
    """A video-shaped placeholder dataset for SDXL frame-level training.

    The dataset returns a stack of frames for one video, one shared video caption,
    and per-frame normalized positions. It can use random tensors or a folder of
    still images while preserving the future real-video dataset interface.
    """

    def __init__(
        self,
        num_videos: int,
        num_frames_per_video: int,
        resolution: int,
        caption: str = "A simple placeholder video caption",
        image_dir: str | Path | None = None,
        frame_sampling: str = "uniform",
        normalize_positions: bool = True,
        seed: int = 0,
    ) -> None:
        if frame_sampling != "uniform":
            raise ValueError(f"Only uniform frame_sampling is implemented, got {frame_sampling!r}.")
        self.num_videos = int(num_videos)
        self.num_frames_per_video = int(num_frames_per_video)
        self.resolution = int(resolution)
        self.caption = caption
        self.frame_sampling = frame_sampling
        self.normalize_positions = normalize_positions
        self.seed = int(seed)

        self.image_paths: list[Path] = []
        if image_dir is not None:
            root = Path(image_dir)
            if root.exists():
                self.image_paths = sorted(
                    path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
                )

    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_positions = make_frame_positions(self.num_frames_per_video, self.normalize_positions)
        if self.image_paths:
            frames = self._load_image_frames(index)
        else:
            generator = torch.Generator().manual_seed(self.seed + int(index))
            frames = torch.rand(
                self.num_frames_per_video,
                3,
                self.resolution,
                self.resolution,
                generator=generator,
                dtype=torch.float32,
            )
            frames = frames * 2.0 - 1.0

        return {
            "frames": frames,
            "caption": self.caption,
            "frame_positions": frame_positions,
        }

    def _load_image_frames(self, index: int) -> torch.Tensor:
        if np is None:
            raise ImportError("numpy is required for placeholder image-folder loading.")
        count = len(self.image_paths)
        if count == 1:
            selected = [0 for _ in range(self.num_frames_per_video)]
        else:
            base = torch.linspace(0, count - 1, steps=self.num_frames_per_video).round().long()
            selected = [int((value + index) % count) for value in base]

        frames = []
        for image_index in selected:
            image = Image.open(self.image_paths[image_index])
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
            array = np.asarray(image).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(tensor)
        return torch.stack(frames, dim=0)


def build_dataset(config: dict[str, Any]) -> Dataset:
    model_config = config["model"]
    data_config = config["data"]
    temporal_config = config.get("temporal_conditioning", {})

    dataset_type = data_config.get("dataset_type", "placeholder")
    if dataset_type != "placeholder":
        raise ValueError(f"Unsupported dataset_type {dataset_type!r}; only 'placeholder' exists in this milestone.")

    return PlaceholderVideoDataset(
        num_videos=data_config.get("placeholder_num_videos", 100),
        num_frames_per_video=data_config["num_frames_per_video"],
        resolution=model_config["resolution"],
        caption=data_config.get("placeholder_caption", "A simple placeholder video caption"),
        image_dir=data_config.get("placeholder_image_dir"),
        frame_sampling=data_config.get("frame_sampling", "uniform"),
        normalize_positions=temporal_config.get("normalize_positions", True),
        seed=config.get("training", {}).get("seed", 0),
    )
