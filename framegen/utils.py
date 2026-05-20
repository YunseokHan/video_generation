from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from PIL import Image
import torch


def set_requires_grad(module: torch.nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def count_trainable_and_frozen(modules: Iterable[torch.nn.Module | None]) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            count = parameter.numel()
            if parameter.requires_grad:
                trainable += count
            else:
                frozen += count
    return trainable, frozen


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.3f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.3f}M"
    if count >= 1_000:
        return f"{count / 1_000:.3f}K"
    return str(count)


def save_image_grid(images: list[Image.Image], output_path: str | Path, columns: int | None = None) -> None:
    if not images:
        raise ValueError("Cannot save a grid with no images.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = columns or len(images)
    columns = max(1, min(columns, len(images)))
    rows = (len(images) + columns - 1) // columns
    width, height = images[0].size
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        grid.paste(image.convert("RGB"), (column * width, row * height))
    grid.save(output_path)


def save_mp4(images: list[Image.Image], output_path: str | Path, fps: int = 8) -> bool:
    try:
        import numpy as np
    except ImportError:
        print("MP4 export failed: numpy is unavailable.")
        return False

    frames = [np.asarray(image.convert("RGB")) for image in images]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, frames, fps=fps)
        return True
    except Exception as imageio_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print(f"MP4 export failed with imageio and ffmpeg was not found: {imageio_error}")
            return False

    frame_array = np.stack(frames, axis=0)
    height, width = frame_array.shape[1:3]
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            input=frame_array.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        print("MP4 export failed: ffmpeg fallback could not be executed.")
        return False
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        print(f"MP4 export failed with ffmpeg fallback: {stderr}")
        return False
    return output_path.exists()
