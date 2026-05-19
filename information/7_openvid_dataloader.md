# OpenVid Dataloader

## Local Dataset Layout

The implemented loader targets:

```text
/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted/
  OpenVid.csv
  panda-ours/*.mp4
  OpenVid_part1/*.mp4
  OpenVid_part2/*.mp4
```

The local scan found:

```text
panda-ours:    32,000 mp4 files
OpenVid_part1: 33,117 mp4 files
OpenVid_part2: 64,220 mp4 files
total:        129,337 mp4 files
```

`OpenVid.csv` has columns including:

```text
source, filepath, video, caption, aesthetic score, motion score,
temporal consistency score, camera motion, frame, fps, seconds
```

The loader uses `filepath` relative to `openvid_root` and `caption` as the
video-level prompt.

## Code

`framegen/data.py` now provides `OpenVidVideoDataset`.

Returned item:

```text
frames:          [F, 3, H, W], float32 in [-1, 1]
caption:         str
frame_positions: [F], normalized to [0, 1] by default
```

`video_collate_fn` stacks these into:

```text
frames:          [B, F, 3, H, W]
captions:        List[str]
frame_positions: [B, F]
```

## Video Decode Backend

The current `video` conda env does not include `imageio`, `cv2`, `decord`,
`av`, `torchvision`, or `pandas`. To avoid adding a Python dependency, the
loader decodes selected frames through the system `ffmpeg` binary.

For each sample it:

1. reads frame count from CSV when available,
2. selects `F` frame indices with `uniform` or `random` sampling,
3. runs an `ffmpeg` `select` filter for those indices,
4. scales and center-crops frames to `model.resolution`,
5. returns a tensor normalized to SDXL VAE input range `[-1, 1]`.

If metadata frame count is missing, it falls back to `ffprobe`.

## Config

Default training config (`configs/train/default.yaml`):

```yaml
data:
  dataset_type: "openvid"
  openvid_root: "/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted"
  openvid_csv: null
  openvid_sources: null
  openvid_max_videos: null
  openvid_min_seconds: 1.0
  openvid_min_frames: 8
  openvid_min_aesthetic_score: null
  openvid_min_motion_score: null
  openvid_camera_motion: null
  openvid_fallback_caption: "A video."
  num_frames_per_video: 8
  frame_sampling: "uniform"
  ffmpeg_path: "ffmpeg"
  ffmpeg_threads: 1
  strict_decode: false
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

Useful filters:

- `openvid_sources`: string or list, for example `["panda-ours"]`
- `openvid_max_videos`: quick subset for smoke runs
- `openvid_min_seconds`: skip very short videos
- `openvid_min_frames`: skip clips with too few frames
- `openvid_min_aesthetic_score`
- `openvid_min_motion_score`
- `openvid_camera_motion`: string or list

`frame_sampling` supports:

- `uniform`
- `random`

For 4-GPU training, the default dataloader launches 4 workers per process, so
up to 16 ffmpeg worker processes can prefetch samples. `ffmpeg_threads: 1`
keeps each subprocess from oversubscribing CPU threads. Pinned memory and
non-blocking `.to(device)` calls are enabled in `train.py` so decoded batches
can overlap host-to-device transfer with GPU work.

## Current Tradeoff

The `ffmpeg` subprocess backend is robust in the current environment but is not
as fast as an in-process video reader. If `decord`, `pyav`, `opencv`, or
`torchvision` is added later, `OpenVidVideoDataset._decode_frames` is the place
to add a faster backend while preserving the same item contract.
