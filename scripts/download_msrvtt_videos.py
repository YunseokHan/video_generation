#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - exercised in user environments.
    raise SystemExit(
        "This script requires the Hugging Face `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc

try:
    from datasets import Video
except ImportError:  # pragma: no cover - older datasets versions can still load metadata.
    Video = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


DEFAULT_DATASET_NAME = "friedrichor/MSR-VTT"
DEFAULT_DATASET_CONFIG = "train_9k"
DEFAULT_SPLIT = "train"
DEFAULT_DATASET_ROOT = "datasets/msrvtt"
DEFAULT_FORMAT_SELECTOR = "bestvideo[ext=mp4]/best[ext=mp4]/bestvideo/best"
EXPECTED_COLUMNS = [
    "video_id",
    "video",
    "caption",
    "source",
    "category",
    "url",
    "start time",
    "end time",
    "id",
]

_STREAM_CACHE: dict[str, str] = {}
_STREAM_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class MSRVTTRecord:
    row_index: int
    sample_id: str
    video_id: str
    url: str
    start_sec: float | None
    end_sec: float | None

    @property
    def duration_sec(self) -> float | None:
        if self.start_sec is None or self.end_sec is None:
            return None
        duration = self.end_sec - self.start_sec
        if duration <= 0:
            return None
        return duration


@dataclass
class DownloadResult:
    row_index: int
    sample_id: str
    video_id: str
    url: str
    start_sec: float | None
    end_sec: float | None
    duration_sec: float | None
    video_path: str
    status: str
    file_size: int = 0
    error: str = ""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the YouTube-backed MSR-VTT train_9k split as video-only mp4 clips. "
            "Completed videos are skipped on rerun, so interrupted jobs can be resumed."
        )
    )
    parser.add_argument("--dataset-name", type=str, default=_env("HF_DATASET_NAME", DEFAULT_DATASET_NAME))
    parser.add_argument("--dataset-config", type=str, default=_env("HF_DATASET_CONFIG", DEFAULT_DATASET_CONFIG))
    parser.add_argument("--split", type=str, default=_env("SPLIT", DEFAULT_SPLIT))
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=_env("DATASET_ROOT", DEFAULT_DATASET_ROOT),
        help="Root directory where videos, manifests, and metadata will be written.",
    )
    parser.add_argument("--videos-subdir", type=str, default=_env("VIDEOS_SUBDIR", "videos"))
    parser.add_argument("--metadata-subdir", type=str, default=_env("METADATA_SUBDIR", "metadata"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=int(_env("NUM_WORKERS", "1") or "1"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-partial-metadata",
        action="store_true",
        help="Write feature metadata even when some videos failed or are missing.",
    )
    parser.add_argument("--yt-dlp-path", type=str, default=_env("YT_DLP_PATH", "yt-dlp"))
    parser.add_argument("--ffmpeg-path", type=str, default=_env("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--ffprobe-path", type=str, default=_env("FFPROBE_PATH", "ffprobe"))
    parser.add_argument(
        "--cookies",
        type=str,
        default=_env("COOKIES"),
        help="Path to a Netscape-format cookies.txt file for yt-dlp.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=_env("COOKIES_FROM_BROWSER"),
        help="Browser spec passed directly to yt-dlp --cookies-from-browser, e.g. 'chrome'.",
    )
    parser.add_argument(
        "--js-runtimes",
        type=str,
        default=_env("JS_RUNTIMES", "node"),
        help="Value passed directly to yt-dlp --js-runtimes.",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=float(_env("SLEEP_REQUESTS", "0.5") or "0.5"),
        help="Sleep interval passed to yt-dlp between internal extractor requests.",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=float(_env("SLEEP_INTERVAL", "2") or "2"),
        help="Base sleep interval passed to yt-dlp before extractor requests.",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=float(_env("MAX_SLEEP_INTERVAL", "4") or "4"),
        help="Maximum randomized sleep interval passed to yt-dlp.",
    )
    parser.add_argument(
        "--format-selector",
        type=str,
        default=_env("FORMAT_SELECTOR", DEFAULT_FORMAT_SELECTOR),
        help="yt-dlp format selector. Defaults to video-only mp4 when possible.",
    )
    parser.add_argument("--video-codec", type=str, default=_env("VIDEO_CODEC", "libx264"))
    parser.add_argument("--crf", type=int, default=int(_env("CRF", "18") or "18"))
    parser.add_argument("--preset", type=str, default=_env("PRESET", "veryfast"))
    parser.add_argument("--min-file-bytes", type=int, default=int(_env("MIN_FILE_BYTES", "1024") or "1024"))
    parser.add_argument("--retries", type=int, default=int(_env("RETRIES", "2") or "2"))
    parser.add_argument("--hf-token", type=str, default=_env("HF_TOKEN"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _resolve_executable(path_or_name: str, *, required: bool = True) -> str | None:
    if os.path.sep in path_or_name:
        path = Path(path_or_name).expanduser().resolve()
        if path.exists():
            return str(path)
        if required:
            raise FileNotFoundError(f"Required executable does not exist: {path}")
        return None
    resolved = shutil.which(path_or_name)
    if resolved is None and required:
        raise FileNotFoundError(
            f"Required executable '{path_or_name}' was not found in PATH. "
            "Install it first or pass an explicit path."
        )
    return resolved


def _normalize_js_runtimes(value: str | None) -> str | None:
    if value == "node":
        node_path = shutil.which("node")
        if node_path:
            return f"node:{node_path}"
    return value


def _run_command(command: list[str], *, verbose: bool = False) -> str:
    if verbose:
        print("RUN:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _feature_to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _feature_to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and ":" in value:
        parts = value.strip().split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60.0 + float(part)
            return total
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(value: str, *, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean[:120] if clean else fallback


def _youtube_url(row_url: str, video_id: str) -> str:
    if row_url:
        return row_url
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    raise ValueError("MSR-VTT row has neither `url` nor `video_id`.")


def _load_msrvtt_dataset(args: argparse.Namespace):
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split,
            token=args.hf_token,
        )
    except TypeError:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split,
            use_auth_token=args.hf_token,
        )
    if "video" in dataset.column_names and Video is not None:
        try:
            dataset = dataset.cast_column("video", Video(decode=False))
        except Exception:
            pass
    return dataset


def _dataset_schema(dataset: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for name, feature in dataset.features.items():
        schema[name] = repr(feature)
    return schema


def _build_records(dataset: Any, *, max_samples: int | None) -> list[MSRVTTRecord]:
    rows = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    columns = set(dataset.column_names)
    video_ids = dataset["video_id"] if "video_id" in columns else [None] * rows
    row_ids = dataset["id"] if "id" in columns else [None] * rows
    urls = dataset["url"] if "url" in columns else [None] * rows
    start_times = dataset["start time"] if "start time" in columns else [None] * rows
    end_times = dataset["end time"] if "end time" in columns else [None] * rows

    records: list[MSRVTTRecord] = []
    for row_index in range(rows):
        video_id = _feature_to_str(video_ids[row_index])
        row_id = _feature_to_str(row_ids[row_index])
        url = _youtube_url(_feature_to_str(urls[row_index]), video_id)
        start_sec = _feature_to_float(start_times[row_index])
        end_sec = _feature_to_float(end_times[row_index])

        stable_id = row_id or video_id or f"row_{row_index:06d}"
        sample_id = f"{row_index:06d}_{_safe_filename(stable_id, fallback=f'row_{row_index:06d}')}"
        records.append(
            MSRVTTRecord(
                row_index=row_index,
                sample_id=sample_id,
                video_id=video_id,
                url=url,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
    return records


def _video_path(videos_dir: Path, record: MSRVTTRecord) -> Path:
    return videos_dir / f"{record.sample_id}.mp4"


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")


def _is_valid_video_file(
    path: Path,
    *,
    min_file_bytes: int,
    ffprobe_path: str | None,
    verbose: bool = False,
) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size < min_file_bytes:
        return False
    if ffprobe_path is None:
        return True
    try:
        output = _run_command(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            verbose=verbose,
        )
        info = json.loads(output) if output else {}
        return bool(info.get("streams"))
    except Exception:
        return False


def _yt_dlp_command_prefix(
    *,
    yt_dlp_path: str,
    cookies: str | None,
    cookies_from_browser: str | None,
    js_runtimes: str | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
) -> list[str]:
    command = [
        yt_dlp_path,
        "--no-playlist",
        "--sleep-requests",
        f"{sleep_requests:.3f}",
        "--sleep-interval",
        f"{sleep_interval:.3f}",
        "--max-sleep-interval",
        f"{max_sleep_interval:.3f}",
    ]
    if cookies is not None:
        command.extend(["--cookies", cookies])
    if cookies_from_browser is not None:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    if js_runtimes is not None:
        command.extend(["--js-runtimes", js_runtimes])
    return command


def resolve_video_stream_url(
    url: str,
    *,
    yt_dlp_path: str,
    cookies: str | None,
    cookies_from_browser: str | None,
    js_runtimes: str | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
    format_selector: str,
    refresh: bool = False,
    verbose: bool = False,
) -> str:
    with _STREAM_CACHE_LOCK:
        if not refresh and url in _STREAM_CACHE:
            return _STREAM_CACHE[url]

    command = _yt_dlp_command_prefix(
        yt_dlp_path=yt_dlp_path,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        js_runtimes=js_runtimes,
        sleep_requests=sleep_requests,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
    )
    command.extend(["-g", "-f", format_selector, url])
    output = _run_command(command, verbose=verbose)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp returned no direct stream URL for {url!r}.")
    stream_url = lines[0]
    with _STREAM_CACHE_LOCK:
        _STREAM_CACHE[url] = stream_url
    return stream_url


def _extract_video_clip(
    *,
    ffmpeg_path: str,
    stream_url: str,
    output_path: Path,
    start_sec: float | None,
    duration_sec: float | None,
    video_codec: str,
    crf: int,
    preset: str,
    verbose: bool,
) -> None:
    temp_path = _temporary_output_path(output_path)
    if temp_path.exists():
        temp_path.unlink()

    command = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None and start_sec > 0:
        command.extend(["-ss", f"{start_sec:.3f}"])
    command.extend(["-i", stream_url])
    if duration_sec is not None:
        command.extend(["-t", f"{duration_sec:.3f}"])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_path),
        ]
    )
    _run_command(command, verbose=verbose)
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced an empty file for {output_path.name}.")
    temp_path.replace(output_path)


def download_record(
    record: MSRVTTRecord,
    *,
    videos_dir: Path,
    dataset_root: Path,
    yt_dlp_path: str,
    ffmpeg_path: str,
    ffprobe_path: str | None,
    cookies: str | None,
    cookies_from_browser: str | None,
    js_runtimes: str | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
    format_selector: str,
    video_codec: str,
    crf: int,
    preset: str,
    min_file_bytes: int,
    retries: int,
    overwrite: bool,
    verbose: bool,
) -> DownloadResult:
    output_path = _video_path(videos_dir, record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_video = str(output_path.relative_to(dataset_root))
    result = DownloadResult(
        row_index=record.row_index,
        sample_id=record.sample_id,
        video_id=record.video_id,
        url=record.url,
        start_sec=record.start_sec,
        end_sec=record.end_sec,
        duration_sec=record.duration_sec,
        video_path=relative_video,
        status="pending",
    )

    temp_path = _temporary_output_path(output_path)
    if temp_path.exists():
        temp_path.unlink()

    if overwrite and output_path.exists():
        output_path.unlink()
    elif _is_valid_video_file(
        output_path,
        min_file_bytes=min_file_bytes,
        ffprobe_path=ffprobe_path,
        verbose=False,
    ):
        result.status = "skipped_existing"
        result.file_size = output_path.stat().st_size
        return result
    elif output_path.exists():
        output_path.unlink()

    last_error = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            stream_url = resolve_video_stream_url(
                record.url,
                yt_dlp_path=yt_dlp_path,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
                js_runtimes=js_runtimes,
                sleep_requests=sleep_requests,
                sleep_interval=sleep_interval,
                max_sleep_interval=max_sleep_interval,
                format_selector=format_selector,
                refresh=attempt > 0,
                verbose=verbose,
            )
            _extract_video_clip(
                ffmpeg_path=ffmpeg_path,
                stream_url=stream_url,
                output_path=output_path,
                start_sec=record.start_sec,
                duration_sec=record.duration_sec,
                video_codec=video_codec,
                crf=crf,
                preset=preset,
                verbose=verbose,
            )
            if not _is_valid_video_file(
                output_path,
                min_file_bytes=min_file_bytes,
                ffprobe_path=ffprobe_path,
                verbose=verbose,
            ):
                raise RuntimeError(f"Output validation failed for {output_path.name}.")
            result.status = "ok"
            result.file_size = output_path.stat().st_size
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if temp_path.exists():
                temp_path.unlink()
            if output_path.exists() and output_path.stat().st_size < min_file_bytes:
                output_path.unlink()

    result.status = "failed"
    result.error = last_error
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.part{path.suffix}")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.part{path.suffix}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _sanitize_feature_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return _sanitize_feature_value(value.item())
        except Exception:
            pass
    if isinstance(value, bytes):
        return {"bytes_omitted": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_feature_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_feature_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_feature_value(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _metadata_rows(
    *,
    dataset: Any,
    records: list[MSRVTTRecord],
    dataset_root: Path,
    videos_dir: Path,
    min_file_bytes: int,
    ffprobe_path: str | None,
) -> Iterable[dict[str, Any]]:
    for record in records:
        video_path = _video_path(videos_dir, record)
        row = dataset[record.row_index]
        features = {key: _sanitize_feature_value(row.get(key)) for key in dataset.column_names}
        yield {
            "row_index": record.row_index,
            "sample_id": record.sample_id,
            "downloaded_video_path": str(video_path.relative_to(dataset_root)),
            "downloaded_video_size": video_path.stat().st_size if video_path.exists() else 0,
            "downloaded_video_valid": _is_valid_video_file(
                video_path,
                min_file_bytes=min_file_bytes,
                ffprobe_path=ffprobe_path,
            ),
            "features": features,
        }


def _progress(iterable: Iterable[Future], *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _print_result(result: DownloadResult) -> None:
    if result.status == "failed":
        print(f"FAILED {result.sample_id}: {result.error}", flush=True)


def _run_downloads_sequential(
    records: list[MSRVTTRecord],
    *,
    videos_dir: Path,
    dataset_root: Path,
    yt_dlp_path: str,
    ffmpeg_path: str,
    ffprobe_path: str | None,
    args: argparse.Namespace,
    js_runtimes: str | None,
) -> tuple[list[DownloadResult], bool]:
    results: list[DownloadResult] = []
    interrupted = False
    iterable: Iterable[MSRVTTRecord]
    iterable = records if tqdm is None else tqdm(records, total=len(records), desc=f"Downloading MSR-VTT {args.split}")
    try:
        for record in iterable:
            result = download_record(
                record,
                videos_dir=videos_dir,
                dataset_root=dataset_root,
                yt_dlp_path=yt_dlp_path,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                js_runtimes=js_runtimes,
                sleep_requests=args.sleep_requests,
                sleep_interval=args.sleep_interval,
                max_sleep_interval=args.max_sleep_interval,
                format_selector=args.format_selector,
                video_codec=args.video_codec,
                crf=args.crf,
                preset=args.preset,
                min_file_bytes=args.min_file_bytes,
                retries=args.retries,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            results.append(result)
            _print_result(result)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Re-run the same command to redownload missing or partial videos.", flush=True)
    return results, interrupted


def _run_downloads_parallel(
    records: list[MSRVTTRecord],
    *,
    videos_dir: Path,
    dataset_root: Path,
    yt_dlp_path: str,
    ffmpeg_path: str,
    ffprobe_path: str | None,
    args: argparse.Namespace,
    js_runtimes: str | None,
) -> tuple[list[DownloadResult], bool]:
    results: list[DownloadResult] = []
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=max(1, args.num_workers))
    futures: dict[Future, MSRVTTRecord] = {}
    try:
        futures = {
            executor.submit(
                download_record,
                record,
                videos_dir=videos_dir,
                dataset_root=dataset_root,
                yt_dlp_path=yt_dlp_path,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                js_runtimes=js_runtimes,
                sleep_requests=args.sleep_requests,
                sleep_interval=args.sleep_interval,
                max_sleep_interval=args.max_sleep_interval,
                format_selector=args.format_selector,
                video_codec=args.video_codec,
                crf=args.crf,
                preset=args.preset,
                min_file_bytes=args.min_file_bytes,
                retries=args.retries,
                overwrite=args.overwrite,
                verbose=args.verbose,
            ): record
            for record in records
        }
        for future in _progress(
            as_completed(futures),
            total=len(futures),
            desc=f"Downloading MSR-VTT {args.split}",
        ):
            result = future.result()
            results.append(result)
            _print_result(result)
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        print("\nInterrupted. Re-run the same command to redownload missing or partial videos.", flush=True)
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=True)
    return results, interrupted


def main() -> None:
    args = parse_args()
    project_root = _repo_root()
    dataset_root = _resolve_path(args.dataset_root, base=project_root)
    videos_dir = dataset_root / args.videos_subdir
    metadata_dir = dataset_root / args.metadata_subdir
    manifests_dir = dataset_root / "manifests"

    yt_dlp_path = _resolve_executable(args.yt_dlp_path, required=True)
    ffmpeg_path = _resolve_executable(args.ffmpeg_path, required=True)
    ffprobe_path = _resolve_executable(args.ffprobe_path, required=False)
    js_runtimes = _normalize_js_runtimes(args.js_runtimes)

    dataset = _load_msrvtt_dataset(args)
    records = _build_records(dataset, max_samples=args.max_samples)
    if not records:
        raise SystemExit("No MSR-VTT records matched the requested split/filter.")

    print(
        f"Loaded {args.dataset_name}/{args.dataset_config} split={args.split} "
        f"rows={len(records)} columns={dataset.column_names}",
        flush=True,
    )
    missing_columns = [name for name in EXPECTED_COLUMNS if name not in dataset.column_names]
    if missing_columns:
        print(f"WARNING: dataset is missing expected columns: {missing_columns}", flush=True)

    if args.num_workers <= 1:
        results, interrupted = _run_downloads_sequential(
            records,
            videos_dir=videos_dir,
            dataset_root=dataset_root,
            yt_dlp_path=yt_dlp_path,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            args=args,
            js_runtimes=js_runtimes,
        )
    else:
        results, interrupted = _run_downloads_parallel(
            records,
            videos_dir=videos_dir,
            dataset_root=dataset_root,
            yt_dlp_path=yt_dlp_path,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            args=args,
            js_runtimes=js_runtimes,
        )

    result_by_index = {result.row_index: result for result in results}
    for record in records:
        if record.row_index not in result_by_index:
            video_path = _video_path(videos_dir, record)
            status = "skipped_existing" if _is_valid_video_file(
                video_path,
                min_file_bytes=args.min_file_bytes,
                ffprobe_path=ffprobe_path,
            ) else "not_attempted"
            result_by_index[record.row_index] = DownloadResult(
                row_index=record.row_index,
                sample_id=record.sample_id,
                video_id=record.video_id,
                url=record.url,
                start_sec=record.start_sec,
                end_sec=record.end_sec,
                duration_sec=record.duration_sec,
                video_path=str(video_path.relative_to(dataset_root)),
                status=status,
                file_size=video_path.stat().st_size if video_path.exists() else 0,
                error="Interrupted before this row was processed." if interrupted else "",
            )
    ordered_results = [result_by_index[record.row_index] for record in records]
    _write_jsonl(manifests_dir / "download_results.jsonl", (asdict(result) for result in ordered_results))

    status_counts: dict[str, int] = {}
    for result in ordered_results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    missing_or_invalid: list[str] = []
    for record in records:
        video_path = _video_path(videos_dir, record)
        if not _is_valid_video_file(
            video_path,
            min_file_bytes=args.min_file_bytes,
            ffprobe_path=ffprobe_path,
        ):
            missing_or_invalid.append(record.sample_id)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "videos_dir": str(videos_dir),
        "metadata_dir": str(metadata_dir),
        "num_records": len(records),
        "status_counts": status_counts,
        "missing_or_invalid_count": len(missing_or_invalid),
        "missing_or_invalid_sample_ids": missing_or_invalid[:50],
        "columns": dataset.column_names,
        "features": _dataset_schema(dataset),
        "cookies": args.cookies,
        "cookies_from_browser": args.cookies_from_browser,
        "js_runtimes": js_runtimes,
        "sleep_requests": float(args.sleep_requests),
        "sleep_interval": float(args.sleep_interval),
        "max_sleep_interval": float(args.max_sleep_interval),
        "format_selector": args.format_selector,
        "video_codec": args.video_codec,
        "interrupted": interrupted,
    }
    _write_json(manifests_dir / "download_summary.json", summary)

    can_write_metadata = not missing_or_invalid or args.allow_partial_metadata
    if can_write_metadata:
        metadata_path = metadata_dir / f"{args.dataset_config}_{args.split}_metadata.jsonl"
        _write_jsonl(
            metadata_path,
            _metadata_rows(
                dataset=dataset,
                records=records,
                dataset_root=dataset_root,
                videos_dir=videos_dir,
                min_file_bytes=args.min_file_bytes,
                ffprobe_path=ffprobe_path,
            ),
        )
        _write_json(
            metadata_dir / f"{args.dataset_config}_{args.split}_dataset_info.json",
            {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "num_records": len(records),
                "columns": dataset.column_names,
                "features": _dataset_schema(dataset),
                "metadata_path": str(metadata_path),
            },
        )
        print(f"Saved feature metadata: {metadata_path}", flush=True)
    else:
        print(
            "Metadata was not written because some videos are missing or invalid. "
            "Re-run the command to resume, or pass --allow-partial-metadata.",
            flush=True,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
