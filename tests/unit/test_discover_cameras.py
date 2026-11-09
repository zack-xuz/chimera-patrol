"""Unit tests for chimera_patrol.ingest.discover / camera (task 2.1)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from chimera_patrol.config import CameraConfig, RoiConfig, VideoSection
from chimera_patrol.ingest.camera import Camera, mask_source
from chimera_patrol.ingest.discover import discover_cameras

TRIANGLE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]


def _video(max_cameras: int = 4) -> VideoSection:
    return VideoSection(max_cameras=max_cameras)


def _roi(roi_id: str = "r1") -> RoiConfig:
    return RoiConfig(id=roi_id, polygon=TRIANGLE)


def test_files_sorted_by_name_and_capped_at_max_cameras() -> None:
    candidates = ["/v/e.mov", "/v/c.mp4", "/v/a.mp4", "/v/d.mkv", "/v/b.avi"]
    cams = discover_cameras(_video(max_cameras=3), [], candidates)
    assert [c.id for c in cams] == ["a", "b", "c"]
    assert all(c.source_type == "file" for c in cams)
    assert cams[0].source == str(Path("/v/a.mp4"))


def test_id_is_file_stem_and_name_defaults_to_id() -> None:
    cams = discover_cameras(_video(), [], ["/v/intrusion_real.mp4", "/v/b.mp4"])
    assert cams[1] == Camera(
        id="intrusion_real",
        name="intrusion_real",
        source_type="file",
        source=str(Path("/v/intrusion_real.mp4")),
        rois=(),
    )


def test_override_with_rtsp_url_switches_source_to_rtsp() -> None:
    cfg = CameraConfig(id="cam1", name="Gate", rtsp_url="rtsp://10.0.0.5:554/s1", rois=[_roi()])
    cams = discover_cameras(_video(), [cfg], ["/v/cam1.mp4", "/v/cam2.mp4"])
    first = cams[0]
    assert first.id == "cam1"
    assert first.name == "Gate"
    assert first.source_type == "rtsp"
    assert first.source == "rtsp://10.0.0.5:554/s1"
    assert first.rois == (_roi(),)
    assert cams[1].source_type == "file"


def test_override_without_rtsp_keeps_file_source_and_applies_name_and_rois() -> None:
    cfg = CameraConfig(id="cam1", name="Dock", rois=[_roi("zone")])
    cams = discover_cameras(_video(), [cfg], ["/v/cam1.mp4", "/v/cam2.mp4"])
    assert cams[0].source_type == "file"
    assert cams[0].name == "Dock"
    assert [r.id for r in cams[0].rois] == ["zone"]


def test_rtsp_only_entry_creates_camera_and_respects_cap() -> None:
    ext = CameraConfig(id="ext", rtsp_url="rtsp://10.0.0.9/stream")
    cams = discover_cameras(_video(max_cameras=2), [ext], [])
    assert [(c.id, c.source_type) for c in cams] == [("ext", "rtsp")]

    capped = discover_cameras(_video(max_cameras=2), [ext], ["/v/a.mp4", "/v/b.mp4"])
    assert [c.id for c in capped] == ["a", "b"]


def test_entry_without_file_or_rtsp_is_ignored() -> None:
    ghost = CameraConfig(id="ghost", name="Ghost")
    cams = discover_cameras(_video(), [ghost], ["/v/a.mp4", "/v/b.mp4"])
    assert [c.id for c in cams] == ["a", "b"]


def test_unknown_extension_ignored_and_extension_case_insensitive() -> None:
    cams = discover_cameras(
        _video(), [], ["/v/notes.txt", "/v/poster.gif", "/v/CLIP.MP4", "/v/b.mkv"]
    )
    assert [c.id for c in cams] == ["CLIP", "b"]


def test_same_stem_with_different_extensions_keeps_one_camera() -> None:
    cams = discover_cameras(_video(), [], ["/v/a.mp4", "/v/a.avi", "/v/b.mp4"])
    assert [(c.id, Path(c.source).name) for c in cams] == [("a", "a.avi"), ("b", "b.mp4")]


def test_directory_scan_uses_direct_files_only(tmp_path: Path) -> None:
    (tmp_path / "x.mp4").write_bytes(b"")
    (tmp_path / "y.mov").write_bytes(b"")
    (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "z.mp4").write_bytes(b"")

    cams = discover_cameras(_video(), [], tmp_path)
    assert [c.id for c in cams] == ["x", "y"]


def test_missing_directory_returns_empty_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        cams = discover_cameras(_video(), [], tmp_path / "missing")
    assert cams == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_fewer_than_two_cameras_logs_warning_and_returns_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        cams = discover_cameras(_video(), [], ["/v/only.mp4"])
    assert [c.id for c in cams] == ["only"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_repr_masks_rtsp_credentials() -> None:
    secret = "s3cret-Pass"
    cam = Camera(
        id="cam1",
        name="cam1",
        source_type="rtsp",
        source=f"rtsp://admin:{secret}@10.0.0.5:554/stream1",
    )
    text = repr(cam)
    assert secret not in text
    assert "admin" not in text
    assert "rtsp://***@10.0.0.5:554/stream1" in text
    assert secret not in str(cam)


def test_repr_of_discovered_rtsp_camera_is_masked() -> None:
    cfg = CameraConfig(id="ext", rtsp_url="rtsp://op:hunter2@10.0.0.9/live")
    (cam,) = discover_cameras(_video(), [cfg], [])
    assert "hunter2" not in repr(cam)
    assert cam.source == "rtsp://op:hunter2@10.0.0.9/live"


def test_mask_source_leaves_plain_urls_and_paths_unchanged() -> None:
    assert mask_source("rtsp://10.0.0.5/stream") == "rtsp://10.0.0.5/stream"
    assert mask_source("/v/a.mp4") == "/v/a.mp4"
