"""Camera 发现：扫描视频文件并与 cameras[] 配置合并（design.md「视频接入」）。

规则摘要：
- 文件扩展名（大小写不敏感）须在 ``video.file_extensions`` 内；按文件名排序取前 ``max_cameras`` 个。
- Camera ID 为文件名主干；``cameras[]`` 中 id 相同的条目覆盖 name、rois，带 ``rtsp_url`` 时改用 RTSP 源。
- ``cameras[]`` 中无对应文件但有 ``rtsp_url`` 的条目创建纯 RTSP Camera，总数仍受 ``max_cameras`` 约束。
- 无文件且无 ``rtsp_url`` 的条目被忽略。
- 可用 Camera 少于 2 路时仅记录 warning，不抛错。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from chimera_patrol.config import CameraConfig, VideoSection
from chimera_patrol.ingest.camera import Camera

log = logging.getLogger(__name__)

MIN_CAMERAS = 2


def discover_cameras(
    video: VideoSection,
    cameras: Sequence[CameraConfig],
    candidates: Path | str | Iterable[Path | str] | None = None,
) -> list[Camera]:
    """返回本次启动可用的 Camera 列表。

    ``candidates`` 为候选文件路径的集合；若传入目录（Path 或 str），则扫描该目录下的直接子文件；
    为 None 时扫描 ``video.source_dir``。
    """
    paths = _resolve_candidates(video, candidates)
    extensions = {_normalize_ext(ext) for ext in video.file_extensions}
    max_cameras = video.max_cameras

    video_files = sorted(
        (Path(p) for p in paths if Path(p).suffix.lower() in extensions),
        key=lambda p: p.name,
    )

    # 同主干不同扩展名（如 a.avi 与 a.mp4）会得到相同 Camera ID，只保留排序靠前的一个。
    file_by_id: dict[str, Path] = {}
    for path in video_files:
        if path.stem in file_by_id:
            log.warning("视频文件 %s 的主干与已选文件重复，已忽略", path.name)
            continue
        file_by_id[path.stem] = path

    overrides = {cfg.id: cfg for cfg in cameras}
    result: list[Camera] = []

    for cam_id, path in list(file_by_id.items())[:max_cameras]:
        cfg = overrides.get(cam_id)
        if cfg is None:
            result.append(Camera(cam_id, cam_id, "file", str(path)))
        elif cfg.rtsp_url:
            result.append(
                Camera(cam_id, cfg.name or cam_id, "rtsp", cfg.rtsp_url, tuple(cfg.rois))
            )
        else:
            result.append(
                Camera(cam_id, cfg.name or cam_id, "file", str(path), tuple(cfg.rois))
            )

    for cfg in cameras:
        if len(result) >= max_cameras:
            break
        if cfg.id in file_by_id or not cfg.rtsp_url:
            continue
        result.append(Camera(cfg.id, cfg.name or cfg.id, "rtsp", cfg.rtsp_url, tuple(cfg.rois)))

    if len(result) < MIN_CAMERAS:
        log.warning("可用 Camera 仅 %d 路，少于 %d 路，仍将启动", len(result), MIN_CAMERAS)
    return result


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def _resolve_candidates(
    video: VideoSection, candidates: Path | str | Iterable[Path | str] | None
) -> list[Path | str]:
    if candidates is None:
        candidates = video.source_dir
    if isinstance(candidates, (str, Path)):
        directory = Path(candidates)
        if not directory.is_dir():
            log.warning("视频目录不存在或不是目录: %s", directory)
            return []
        return [entry for entry in directory.iterdir() if entry.is_file()]
    return list(candidates)
