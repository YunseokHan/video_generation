from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_env_file(path: str | Path | None, override: bool = False) -> bool:
    if path is None:
        apply_hf_env_aliases()
        return False

    env_path = Path(path)
    loaded = False
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            loaded = bool(load_dotenv(env_path, override=override))
        except ImportError:
            loaded = _load_env_file_without_dependency(env_path, override=override)

    apply_hf_env_aliases()
    return loaded


def _load_env_file_without_dependency(path: Path, override: bool = False) -> bool:
    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_env_value(value.strip())
        if override or key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def apply_hf_env_aliases() -> None:
    hf_cache = _get_nonempty_env("HF_CACHE")
    if hf_cache:
        cache_path = Path(hf_cache)
        hub_cache = cache_path if cache_path.name == "hub" else cache_path / "hub"
        os.environ.setdefault("HF_HOME", str(cache_path))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("DIFFUSERS_CACHE", str(hub_cache))

    hf_token = _get_nonempty_env("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)


def get_hf_cache_dir(model_config: dict[str, Any] | None = None) -> str | None:
    if model_config:
        cache_dir = model_config.get("cache_dir")
        if cache_dir:
            return str(cache_dir)
    return (
        _get_nonempty_env("HUGGINGFACE_HUB_CACHE")
        or _get_nonempty_env("HF_HUB_CACHE")
        or _hf_cache_to_hub_cache(_get_nonempty_env("HF_CACHE"))
    )


def get_hf_token(model_config: dict[str, Any] | None = None) -> str | None:
    if model_config:
        token = model_config.get("token")
        if token:
            return str(token)
    return (
        _get_nonempty_env("HF_TOKEN")
        or _get_nonempty_env("HUGGING_FACE_HUB_TOKEN")
        or _get_nonempty_env("HUGGINGFACE_TOKEN")
    )


def _hf_cache_to_hub_cache(hf_cache: str | None) -> str | None:
    if not hf_cache:
        return None
    cache_path = Path(hf_cache)
    return str(cache_path if cache_path.name == "hub" else cache_path / "hub")


def _get_nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value
