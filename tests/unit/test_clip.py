"""Unit tests for chimera_patrol.evidence.clip (task 5.4)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from chimera_patrol.config import EvidenceSection
from chimera_patrol.evidence import clip as clip_mod
from chimera_patrol.evidence.clip import clip_window, save_clip_safely, write_clip
from chimera_patrol.evidence.ring_buffer import RingBuffer


def _bgr_frame(width: int, height: int, value: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = value  # B
    frame[..., 1] = np.arange(width, dtype=np.uint8)[None, :]  # G gradient
    frame[..., 2] = (255 - value) % 256  # R
    return frame


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _read_back(path: Path) -> tuple[tuple[int, int], int, str]:
    """Return ((width, height), decoded frame count, codec name) using imageio-ffmpeg's reader."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    frame_count = sum(1 for _ in reader)
    return tuple(meta["size"]), frame_count, meta["codec"]


def _pix_fmt(path: Path) -> str:
    """Read the output pixel format from the bundled ffmpeg's stream info (ffprobe-style check)."""
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stderr.splitlines():
        if "Video:" in line:
            return "yuv420p" if "yuv420p" in line else line
    raise AssertionError(f"no video stream reported for {path}")


def _filled_ring(seconds: float, fps: int, width: int = 64, height: int = 48) -> RingBuffer:
    ring = RingBuffer(window_s=seconds + 10, max_fps=fps)
    step = 1000 // fps
    for i in range(int(seconds * fps) + 1):
        ts = i * step
        ring.push(ts, _jpeg(_bgr_frame(width, height, value=i % 256)))
    return ring


def test_write_clip_produces_h264_yuv420p_with_expected_size_and_frames(tmp_path: Path) -> None:
    frames = [_bgr_frame(64, 48, value=i * 10) for i in range(12)]
    out = write_clip(frames, fps=10, out_path=tmp_path / "clip.mp4")

    assert out.is_file() and out.stat().st_size > 0
    size, count, codec = _read_back(out)
    assert size == (64, 48)
    assert count == 12
    assert codec == "h264"
    assert _pix_fmt(out) == "yuv420p"


def test_write_clip_crops_odd_dimensions_to_even(tmp_path: Path) -> None:
    frames = [_bgr_frame(65, 47, value=i) for i in range(5)]
    out = write_clip(iter(frames), fps=10, out_path=tmp_path / "odd.mp4")

    size, count, _ = _read_back(out)
    assert size == (64, 46)  # one pixel removed from each odd side, no upscaling
    assert count == 5
    assert _pix_fmt(out) == "yuv420p"


def test_write_clip_rejects_empty_and_mixed_size_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_clip([], fps=10, out_path=tmp_path / "empty.mp4")

    mixed = [_bgr_frame(64, 48, 0), _bgr_frame(32, 32, 0)]
    with pytest.raises(ValueError):
        write_clip(mixed, fps=10, out_path=tmp_path / "mixed.mp4")


def test_clip_window_returns_inclusive_ascending_range() -> None:
    ring = _filled_ring(seconds=10, fps=10)  # frames at 0, 100, ..., 10000 ms
    records = clip_window(ring, end_ms=5000, pre_s=1, post_s=1)

    timestamps = [ts for ts, _ in records]
    assert timestamps == list(range(4000, 6001, 100))
    assert timestamps == sorted(timestamps)


def test_save_clip_safely_writes_ready_clip(tmp_path: Path) -> None:
    ring = _filled_ring(seconds=10, fps=10)
    evidence = EvidenceSection(base_dir=tmp_path, clip_pre_s=1, clip_post_s=1, clip_fps=10)
    expected_frames = len(clip_window(ring, 5000, 1, 1))

    status, rel = save_clip_safely(
        ring, evidence=evidence, camera_id="cam1", event_id="evt_1", end_ms=5000
    )

    assert status == "ready"
    assert rel == "cam1/evt_1/clip.mp4"
    out = tmp_path / rel
    assert out.is_file()
    assert not (tmp_path / "cam1" / "evt_1" / "clip.partial.mp4").exists()
    size, count, _ = _read_back(out)
    assert size == (64, 48)
    assert count == expected_frames == 21


def test_save_clip_safely_returns_failed_and_logs_on_forced_encode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ring = _filled_ring(seconds=10, fps=10)
    evidence = EvidenceSection(base_dir=tmp_path, clip_pre_s=1, clip_post_s=1, clip_fps=10)

    def _boom(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("forced encoder failure")

    monkeypatch.setattr(clip_mod, "write_clip", _boom)
    with caplog.at_level(logging.ERROR, logger="chimera_patrol.evidence.clip"):
        status, message = save_clip_safely(
            ring, evidence=evidence, camera_id="cam9", event_id="evt_fail", end_ms=5000
        )

    assert status == "failed"
    assert message == "RuntimeError: forced encoder failure"
    assert not (tmp_path / "cam9" / "evt_fail" / "clip.mp4").exists()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "event_id=evt_fail" in errors[0].getMessage()
    assert "camera_id=cam9" in errors[0].getMessage()


def test_save_clip_safely_fails_without_raising_when_window_is_empty(tmp_path: Path) -> None:
    ring = RingBuffer(window_s=20, max_fps=10)  # nothing stored
    evidence = EvidenceSection(base_dir=tmp_path)

    status, message = save_clip_safely(
        ring, evidence=evidence, camera_id="cam1", event_id="evt_empty", end_ms=1000
    )

    assert status == "failed"
    assert "no buffered frames" in message


def test_save_clip_safely_rejects_unsafe_path_components(tmp_path: Path) -> None:
    ring = _filled_ring(seconds=10, fps=10)
    evidence = EvidenceSection(base_dir=tmp_path, clip_pre_s=1, clip_post_s=1)

    status, message = save_clip_safely(
        ring, evidence=evidence, camera_id="..", event_id="evt", end_ms=5000
    )

    assert status == "failed"
    assert "camera_id" in message
