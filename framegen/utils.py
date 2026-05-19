from __future__ import annotations

from pathlib import Path
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
        import imageio.v3 as iio
        import numpy as np
    except ImportError:
        return False

    frames = [np.asarray(image.convert("RGB")) for image in images]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(output_path, frames, fps=fps)
    except Exception:
        return False
    return True
