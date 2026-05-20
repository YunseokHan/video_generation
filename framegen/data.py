from __future__ import annotations

import csv
import math
import shutil
import subprocess
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
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
DEFAULT_OPENVID_ROOT = "/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted"
PIL_INTERPOLATION = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
FFMPEG_SCALE_FLAGS = {
    "nearest": "neighbor",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "lanczos": "lanczos",
}


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
    def metadata_tensor(item: dict[str, Any], key: str, default: tuple[int, int]) -> torch.Tensor:
        value = item.get(key, default)
        if isinstance(value, torch.Tensor):
            return value.long()
        return torch.tensor(value, dtype=torch.long)

    first_resolution = int(items[0]["frames"].shape[-1])
    return {
        "frames": torch.stack([item["frames"] for item in items], dim=0),
        "captions": [item["caption"] for item in items],
        "frame_positions": torch.stack([item["frame_positions"] for item in items], dim=0),
        "original_sizes": torch.stack(
            [
                metadata_tensor(item, "original_size", (first_resolution, first_resolution))
                for item in items
            ],
            dim=0,
        ),
        "crop_top_lefts": torch.stack(
            [metadata_tensor(item, "crop_top_left", (0, 0)) for item in items],
            dim=0,
        ),
        "target_sizes": torch.stack(
            [
                metadata_tensor(item, "target_size", (first_resolution, first_resolution))
                for item in items
            ],
            dim=0,
        ),
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
        center_crop: bool = False,
        random_flip: bool = False,
        image_interpolation_mode: str = "lanczos",
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
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.interpolation = PIL_INTERPOLATION.get(
            str(image_interpolation_mode).lower(),
            Image.Resampling.LANCZOS,
        )
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
            "original_size": (self.resolution, self.resolution),
            "crop_top_left": (0, 0),
            "target_size": (self.resolution, self.resolution),
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
        do_flip = bool(self.random_flip and torch.rand(()).item() < 0.5)
        for image_index in selected:
            image = Image.open(self.image_paths[image_index])
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((self.resolution, self.resolution), self.interpolation)
            if do_flip:
                image = ImageOps.mirror(image)
            array = np.asarray(image).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(tensor)
        return torch.stack(frames, dim=0)


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


class OpenVidVideoDataset(Dataset):
    """OpenVid CSV + mp4 dataloader.

    Expected extracted layout:

    ```text
    OpenVid_extracted/
      OpenVid.csv
      panda-ours/*.mp4
      OpenVid_part1/*.mp4
      OpenVid_part2/*.mp4
    ```

    Video frames are decoded through the system `ffmpeg` binary so the dataset
    works in the current `video` conda env without extra Python video packages.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_OPENVID_ROOT,
        csv_path: str | Path | None = None,
        num_frames_per_video: int = 8,
        resolution: int = 1024,
        frame_sampling: str = "uniform",
        normalize_positions: bool = True,
        max_videos: int | None = None,
        sources: list[str] | str | None = None,
        min_seconds: float | None = None,
        min_frames: int | None = None,
        min_aesthetic_score: float | None = None,
        min_motion_score: float | None = None,
        camera_motion: list[str] | str | None = None,
        fallback_caption: str = "A video.",
        ffmpeg_path: str = "ffmpeg",
        ffmpeg_threads: int = 1,
        strict_decode: bool = False,
        center_crop: bool = False,
        random_flip: bool = False,
        image_interpolation_mode: str = "lanczos",
    ) -> None:
        if frame_sampling not in {"uniform", "random"}:
            raise ValueError(f"OpenVid frame_sampling must be 'uniform' or 'random', got {frame_sampling!r}.")
        self.root = Path(root)
        self.csv_path = Path(csv_path) if csv_path is not None else self.root / "OpenVid.csv"
        self.num_frames_per_video = int(num_frames_per_video)
        self.resolution = int(resolution)
        self.frame_sampling = frame_sampling
        self.normalize_positions = bool(normalize_positions)
        self.fallback_caption = fallback_caption
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_threads = max(1, int(ffmpeg_threads))
        self.strict_decode = bool(strict_decode)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.ffmpeg_scale_flags = FFMPEG_SCALE_FLAGS.get(
            str(image_interpolation_mode).lower(),
            "lanczos",
        )
        self._dimension_cache: dict[Path, tuple[int, int]] = {}

        if self.num_frames_per_video <= 0:
            raise ValueError("num_frames_per_video must be positive.")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"OpenVid metadata CSV not found: {self.csv_path}")
        if shutil.which(self.ffmpeg_path) is None:
            raise FileNotFoundError(
                f"ffmpeg binary {self.ffmpeg_path!r} was not found. Install ffmpeg or set data.ffmpeg_path."
            )

        self.samples = self._load_samples(
            max_videos=max_videos,
            sources=set(_as_list(sources) or []),
            min_seconds=min_seconds,
            min_frames=min_frames,
            min_aesthetic_score=min_aesthetic_score,
            min_motion_score=min_motion_score,
            camera_motion=set(_as_list(camera_motion) or []),
        )
        if not self.samples:
            raise ValueError(
                "No OpenVid samples matched the configured filters. "
                f"root={self.root}, csv={self.csv_path}"
            )

    def _load_samples(
        self,
        max_videos: int | None,
        sources: set[str],
        min_seconds: float | None,
        min_frames: int | None,
        min_aesthetic_score: float | None,
        min_motion_score: float | None,
        camera_motion: set[str],
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = row.get("source", "")
                if sources and source not in sources:
                    continue
                if camera_motion and row.get("camera motion", "") not in camera_motion:
                    continue

                frame_count = _optional_int(row.get("frame"))
                seconds = _optional_float(row.get("seconds"))
                if min_frames is not None and (frame_count is None or frame_count < min_frames):
                    continue
                if min_seconds is not None and (seconds is None or seconds < min_seconds):
                    continue
                aesthetic = _optional_float(row.get("aesthetic score"))
                if min_aesthetic_score is not None and (aesthetic is None or aesthetic < min_aesthetic_score):
                    continue
                motion = _optional_float(row.get("motion score"))
                if min_motion_score is not None and (motion is None or motion < min_motion_score):
                    continue

                relative_path = row.get("filepath") or (
                    f"{source}/{row['video']}" if source and row.get("video") else row.get("video", "")
                )
                path = self.root / relative_path
                if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.exists():
                    continue

                caption = row.get("caption") or self.fallback_caption
                samples.append(
                    {
                        "path": path,
                        "caption": caption,
                        "frame_count": frame_count,
                        "seconds": seconds,
                        "fps": _optional_float(row.get("fps")),
                        "source": source,
                    }
                )
                if max_videos is not None and len(samples) >= int(max_videos):
                    break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index) % len(self.samples)]
        frame_count = sample.get("frame_count")
        if frame_count is None or frame_count <= 0:
            frame_count = self._probe_frame_count(sample["path"])
        frame_indices = self._sample_frame_indices(frame_count)
        transform = self._sample_spatial_transform(sample["path"])
        frames = self._decode_frames(sample["path"], frame_indices, transform)
        frame_positions = self._make_positions(frame_indices, frame_count)
        return {
            "frames": frames,
            "caption": sample["caption"],
            "frame_positions": frame_positions,
            "original_size": transform["original_size"],
            "crop_top_left": transform["crop_top_left"],
            "target_size": transform["target_size"],
        }

    def _probe_frame_count(self, path: Path) -> int:
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            raise RuntimeError(f"Missing frame count for {path} and ffprobe is unavailable.")
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        frame_count = _optional_int(result.stdout.strip())
        if result.returncode != 0 or frame_count is None or frame_count <= 0:
            raise RuntimeError(f"Could not probe frame count for {path}: {result.stderr.strip()}")
        return frame_count

    def _probe_video_dimensions(self, path: Path) -> tuple[int, int]:
        cached = self._dimension_cache.get(path)
        if cached is not None:
            return cached
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            self._dimension_cache[path] = (self.resolution, self.resolution)
            return self._dimension_cache[path]
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        width = height = None
        if result.returncode == 0 and "x" in result.stdout:
            width_text, height_text = result.stdout.strip().split("x", maxsplit=1)
            width = _optional_int(width_text)
            height = _optional_int(height_text)
        if width is None or height is None or width <= 0 or height <= 0:
            width = height = self.resolution
        self._dimension_cache[path] = (int(width), int(height))
        return self._dimension_cache[path]

    def _sample_spatial_transform(self, path: Path) -> dict[str, Any]:
        width, height = self._probe_video_dimensions(path)
        scale = self.resolution / max(min(width, height), 1)
        scaled_width = max(self.resolution, int(math.ceil(width * scale)))
        scaled_height = max(self.resolution, int(math.ceil(height * scale)))
        max_x = max(scaled_width - self.resolution, 0)
        max_y = max(scaled_height - self.resolution, 0)
        if self.center_crop:
            crop_x = max_x // 2
            crop_y = max_y // 2
        else:
            crop_x = int(torch.randint(0, max_x + 1, ()).item()) if max_x > 0 else 0
            crop_y = int(torch.randint(0, max_y + 1, ()).item()) if max_y > 0 else 0
        hflip = bool(self.random_flip and torch.rand(()).item() < 0.5)
        return {
            "original_size": (int(height), int(width)),
            "scaled_size": (int(scaled_height), int(scaled_width)),
            "crop_top_left": (int(crop_y), int(crop_x)),
            "target_size": (self.resolution, self.resolution),
            "hflip": hflip,
        }

    def _sample_frame_indices(self, frame_count: int) -> list[int]:
        usable_count = max(int(frame_count), 1)
        if self.frame_sampling == "uniform":
            positions = torch.linspace(0, usable_count - 1, steps=self.num_frames_per_video)
        else:
            positions = torch.rand(self.num_frames_per_video) * max(usable_count - 1, 1)
            positions, _ = torch.sort(positions)
        return [max(0, min(usable_count - 1, int(round(float(position))))) for position in positions]

    def _make_positions(self, frame_indices: list[int], frame_count: int) -> torch.Tensor:
        positions = torch.tensor(frame_indices, dtype=torch.float32)
        if self.normalize_positions:
            return positions / max(int(frame_count) - 1, 1)
        return positions

    def _decode_frames(
        self,
        path: Path,
        frame_indices: list[int],
        transform: dict[str, Any],
    ) -> torch.Tensor:
        unique_indices = sorted(set(frame_indices))
        select_expression = "+".join(f"eq(n,{index})" for index in unique_indices)
        scaled_height, scaled_width = transform["scaled_size"]
        crop_y, crop_x = transform["crop_top_left"]
        filters = [
            f"select='{select_expression}'",
            f"scale={scaled_width}:{scaled_height}:flags={self.ffmpeg_scale_flags}",
        ]
        if transform.get("hflip", False):
            filters.append("hflip")
        filters.append(f"crop={self.resolution}:{self.resolution}:{crop_x}:{crop_y}")
        filters.append("setsar=1")
        video_filter = (
            ",".join(filters)
        )
        command = [
            self.ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            str(self.ffmpeg_threads),
            "-i",
            str(path),
            "-vf",
            video_filter,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        frame_bytes = self.resolution * self.resolution * 3
        decoded_count = len(result.stdout) // frame_bytes
        if result.returncode != 0 or decoded_count <= 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            if self.strict_decode:
                raise RuntimeError(f"ffmpeg failed for {path}: {message}")
            return torch.zeros(self.num_frames_per_video, 3, self.resolution, self.resolution)

        usable_bytes = decoded_count * frame_bytes
        array = torch.frombuffer(bytearray(result.stdout[:usable_bytes]), dtype=torch.uint8)
        frames = array.reshape(decoded_count, self.resolution, self.resolution, 3)
        frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if decoded_count < len(unique_indices):
            pad = frames[-1:].expand(len(unique_indices) - decoded_count, -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)

        frame_by_index = {
            frame_index: frames[position]
            for position, frame_index in enumerate(unique_indices[: frames.shape[0]])
        }
        selected_frames = [frame_by_index.get(frame_index, frames[-1]) for frame_index in frame_indices]
        return torch.stack(selected_frames, dim=0)


def build_dataset(config: dict[str, Any]) -> Dataset:
    model_config = config["model"]
    data_config = config["data"]
    temporal_config = config.get("temporal_conditioning", {})

    dataset_type = data_config.get("dataset_type", "placeholder")
    if dataset_type == "placeholder":
        return PlaceholderVideoDataset(
            num_videos=data_config.get("placeholder_num_videos", 100),
            num_frames_per_video=data_config["num_frames_per_video"],
            resolution=model_config["resolution"],
            caption=data_config.get("placeholder_caption", "A simple placeholder video caption"),
            image_dir=data_config.get("placeholder_image_dir"),
            frame_sampling=data_config.get("frame_sampling", "uniform"),
            normalize_positions=temporal_config.get("normalize_positions", True),
            center_crop=data_config.get("center_crop", False),
            random_flip=data_config.get("random_flip", False),
            image_interpolation_mode=data_config.get("image_interpolation_mode", "lanczos"),
            seed=config.get("training", {}).get("seed", 0),
        )
    if dataset_type == "openvid":
        return OpenVidVideoDataset(
            root=data_config.get("openvid_root", DEFAULT_OPENVID_ROOT),
            csv_path=data_config.get("openvid_csv"),
            num_frames_per_video=data_config["num_frames_per_video"],
            resolution=model_config["resolution"],
            frame_sampling=data_config.get("frame_sampling", "uniform"),
            normalize_positions=temporal_config.get("normalize_positions", True),
            max_videos=data_config.get("openvid_max_videos"),
            sources=data_config.get("openvid_sources"),
            min_seconds=data_config.get("openvid_min_seconds"),
            min_frames=data_config.get("openvid_min_frames"),
            min_aesthetic_score=data_config.get("openvid_min_aesthetic_score"),
            min_motion_score=data_config.get("openvid_min_motion_score"),
            camera_motion=data_config.get("openvid_camera_motion"),
            fallback_caption=data_config.get("openvid_fallback_caption", "A video."),
            ffmpeg_path=data_config.get("ffmpeg_path", "ffmpeg"),
            ffmpeg_threads=data_config.get("ffmpeg_threads", 1),
            strict_decode=data_config.get("strict_decode", False),
            center_crop=data_config.get("center_crop", False),
            random_flip=data_config.get("random_flip", False),
            image_interpolation_mode=data_config.get("image_interpolation_mode", "lanczos"),
        )
    raise ValueError(f"Unsupported dataset_type {dataset_type!r}; expected 'placeholder' or 'openvid'.")
