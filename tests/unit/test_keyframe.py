"""Unit tests for chimera_patrol.evidence.keyframe (task 5.3)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from chimera_patrol.evidence.keyframe import KeyframeUnavailable, save_keyframe
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.rules.types import OverlayItem

_JPEG_MAGIC = b"\xff\xd8\xff"


def _frame_jpeg(value: int) -> bytes:
    """A solid-colour 160x120 JPEG. Different ``value`` gives different bytes."""
    img = np.full((120, 160, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def _ring_with_frames(timestamps_ms: list[int]) -> tuple[RingBuffer, dict[int, bytes]]:
    """Push one frame per timestamp. The value of each frame encodes its index."""
    ring = RingBuffer(window_s=60, max_fps=10)
    frames: dict[int, bytes] = {}
    for i, ts in enumerate(timestamps_ms):
        data = _frame_jpeg(10 + i * 20)
        assert ring.push(ts, data) is True
        frames[ts] = data
    return ring, frames


def _save(
    ring: RingBuffer,
    base: Path,
    start_ms: int,
    end_ms: int,
    overlay: Sequence[OverlayItem] = (),
) -> tuple[str, str]:
    return save_keyframe(
        ring,
        start_ms=start_ms,
        end_ms=end_ms,
        overlay=list(overlay),
        base_dir=base,
        jpeg_quality=90,
        camera_id="cam_a",
        event_id="evt-1",
    )


def test_selects_frame_nearest_to_midpoint(tmp_path: Path) -> None:
    ring, frames = _ring_with_frames([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # Window [310, 690] has midpoint 500, and the frame at 500 is the closest.
    _save(ring, tmp_path, 310, 690)
    raw = (tmp_path / "cam_a" / "evt-1" / "keyframe_raw.jpg").read_bytes()
    assert raw == frames[500]


def test_falls_back_to_nearest_frame_in_whole_buffer(tmp_path: Path) -> None:
    ring, frames = _ring_with_frames([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # No frame lies inside [2000, 2400]; midpoint 2200 maps to the newest frame at 1000.
    _save(ring, tmp_path, 2000, 2400)
    raw = (tmp_path / "cam_a" / "evt-1" / "keyframe_raw.jpg").read_bytes()
    assert raw == frames[1000]


def test_annotated_differs_from_raw_when_overlay_present(tmp_path: Path) -> None:
    ring, _ = _ring_with_frames([0, 100, 200])
    overlay = [
        OverlayItem(
            box=(20.0, 20.0, 100.0, 100.0),
            label="person",
            conf=0.91,
            roi_polygon=[(10.0, 10.0), (150.0, 10.0), (150.0, 110.0), (10.0, 110.0)],
        )
    ]
    annotated_rel, raw_rel = _save(ring, tmp_path, 0, 200, overlay)
    annotated = (tmp_path / annotated_rel).read_bytes()
    raw = (tmp_path / raw_rel).read_bytes()
    assert annotated != raw
    decoded = cv2.imdecode(np.frombuffer(annotated, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # The box edge is drawn in green (BGR 0,255,0), so the image has green-ish pixels.
    assert (decoded[:, :, 1].astype(int) - decoded[:, :, 0].astype(int) > 100).any()


def test_directory_layout_and_relative_paths(tmp_path: Path) -> None:
    ring, _ = _ring_with_frames([0, 100])
    annotated_rel, raw_rel = _save(ring, tmp_path, 0, 100)
    assert annotated_rel == "cam_a/evt-1/keyframe_annotated.jpg"
    assert raw_rel == "cam_a/evt-1/keyframe_raw.jpg"
    assert "\\" not in annotated_rel and "\\" not in raw_rel
    out_dir = tmp_path / "cam_a" / "evt-1"
    assert (out_dir / "keyframe_annotated.jpg").read_bytes()[:3] == _JPEG_MAGIC
    assert (out_dir / "keyframe_raw.jpg").read_bytes()[:3] == _JPEG_MAGIC


def test_non_jpeg_frame_is_reencoded_as_jpeg(tmp_path: Path) -> None:
    ring = RingBuffer(window_s=60, max_fps=10)
    img = np.full((120, 160, 3), 80, dtype=np.uint8)
    ok, png = cv2.imencode(".png", img)
    assert ok
    ring.push(0, png.tobytes())
    _, raw_rel = _save(ring, tmp_path, 0, 0)
    raw = (tmp_path / raw_rel).read_bytes()
    assert raw[:3] == _JPEG_MAGIC


def test_no_frames_raises_keyframe_unavailable(tmp_path: Path) -> None:
    ring = RingBuffer(window_s=60, max_fps=10)
    with pytest.raises(KeyframeUnavailable):
        _save(ring, tmp_path, 0, 1000)
    assert not (tmp_path / "cam_a" / "evt-1").exists()


def test_undecodable_frame_raises_keyframe_unavailable(tmp_path: Path) -> None:
    ring = RingBuffer(window_s=60, max_fps=10)
    ring.push(0, b"not an image at all")
    with pytest.raises(KeyframeUnavailable):
        _save(ring, tmp_path, 0, 0)
