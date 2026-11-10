"""关键帧保存（design.md「证据截取」4.1，任务 5.3）。

从 RingBuffer 中选取事件中点附近的帧，保存两张 JPEG：

- ``keyframe_raw.jpg``：原图（缓存中的 JPEG 字节，非 JPEG 时重新编码）。
- ``keyframe_annotated.jpg``：在原图上绘制 ``OverlayItem`` 的检测框、label、conf 与 ROI 多边形。

存储路径为 ``{base_dir}/{camera_id}/{event_id}/``，返回相对 ``base_dir`` 的路径（正斜杠）。
本模块不依赖配置，所有参数由调用方传入。
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from chimera_patrol.evidence.ring_buffer import Frame, RingBuffer, select_frame
from chimera_patrol.rules.types import OverlayItem

__all__ = ["KeyframeUnavailable", "save_keyframe", "save_keyframe_from_jpeg"]

ANNOTATED_NAME = "keyframe_annotated.jpg"
RAW_NAME = "keyframe_raw.jpg"

_JPEG_MAGIC = b"\xff\xd8\xff"
_BOX_COLOR = (0, 255, 0)  # BGR
_ROI_COLOR = (255, 200, 0)  # BGR
_TEXT_COLOR = (0, 255, 0)  # BGR


class KeyframeUnavailable(Exception):
    """无法生成或保存关键帧（缓存无帧、帧无法解码、写盘失败）。调用方应将证据标记为失败。"""


def _check_path_segment(name: str, value: str) -> None:
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{name} 不能为空、'.'、'..' 或包含路径分隔符: {value!r}")


def _select(ring: RingBuffer, start_ms: int, end_ms: int) -> Frame:
    mid_ms = (start_ms + end_ms) / 2
    window = ring.snapshot(start_ms, end_ms)
    if not window:
        # 事件区间内没有帧时，退回到整个缓存中最接近中点的帧。
        window = ring.snapshot(-sys.maxsize, sys.maxsize)
    if not window:
        raise KeyframeUnavailable("环形缓存中没有可用帧")
    return select_frame(window, mid_ms)


def _draw(image: np.ndarray, overlay: Sequence[OverlayItem]) -> None:
    for item in overlay:
        x1, y1, x2, y2 = (int(round(v)) for v in item.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        text = item.label if item.conf is None else f"{item.label} {item.conf:.2f}"
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        if item.roi_polygon:
            pts = np.array(
                [(int(round(x)), int(round(y))) for x, y in item.roi_polygon], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, _ROI_COLOR, 2)


def _encode(image: np.ndarray, jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise KeyframeUnavailable("JPEG 编码失败")
    return buf.tobytes()


def save_keyframe(
    ring: RingBuffer,
    *,
    start_ms: int,
    end_ms: int,
    overlay: Sequence[OverlayItem],
    base_dir: Path,
    jpeg_quality: int,
    camera_id: str,
    event_id: str,
) -> tuple[str, str]:
    """保存事件关键帧的标注版与原图版，返回 ``(annotated_rel, raw_rel)``。

    两个路径都相对 ``base_dir``，使用正斜杠，便于写入 EventRepository。
    缓存无帧、帧无法解码或写盘失败时抛出 ``KeyframeUnavailable``。
    ``camera_id`` 或 ``event_id`` 含路径分隔符等非法字符时抛出 ``ValueError``。
    """
    _check_path_segment("camera_id", camera_id)
    _check_path_segment("event_id", event_id)

    _, frame_bytes = _select(ring, start_ms, end_ms)

    decoded = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise KeyframeUnavailable("选中的缓存帧无法解码为图像")

    raw_bytes = frame_bytes if frame_bytes[:3] == _JPEG_MAGIC else _encode(decoded, jpeg_quality)

    annotated = decoded.copy()
    _draw(annotated, overlay)
    annotated_bytes = _encode(annotated, jpeg_quality)

    rel_dir = f"{camera_id}/{event_id}"
    out_dir = base_dir / camera_id / event_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / RAW_NAME).write_bytes(raw_bytes)
        (out_dir / ANNOTATED_NAME).write_bytes(annotated_bytes)
    except OSError as exc:
        raise KeyframeUnavailable(f"关键帧写入失败: {exc}") from exc

    return f"{rel_dir}/{ANNOTATED_NAME}", f"{rel_dir}/{RAW_NAME}"


def save_keyframe_from_jpeg(
    jpeg: bytes,
    *,
    overlay: Sequence[OverlayItem],
    base_dir: Path,
    jpeg_quality: int,
    camera_id: str,
    event_id: str,
) -> tuple[str, str]:
    """用给定的 JPEG 字节保存关键帧（SOP 事件使用采样帧）。返回 ``(annotated_rel, raw_rel)``。

    语义与 ``save_keyframe`` 一致：原图版保存输入字节，标注版在解码后的副本上绘制 overlay。
    解码失败或写盘失败时抛出 ``KeyframeUnavailable``。
    """
    _check_path_segment("camera_id", camera_id)
    _check_path_segment("event_id", event_id)

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise KeyframeUnavailable("给定的关键帧字节无法解码为图像")

    raw_bytes = jpeg if jpeg[:3] == _JPEG_MAGIC else _encode(decoded, jpeg_quality)
    annotated = decoded.copy()
    _draw(annotated, overlay)
    annotated_bytes = _encode(annotated, jpeg_quality)

    rel_dir = f"{camera_id}/{event_id}"
    out_dir = base_dir / camera_id / event_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / RAW_NAME).write_bytes(raw_bytes)
        (out_dir / ANNOTATED_NAME).write_bytes(annotated_bytes)
    except OSError as exc:
        raise KeyframeUnavailable(f"关键帧写入失败: {exc}") from exc

    return f"{rel_dir}/{ANNOTATED_NAME}", f"{rel_dir}/{RAW_NAME}"
