"""复核与恢复的集成测试（任务 10.7、10.8、10.9）：慢 VLM 与故障隔离、启动恢复接线。

使用假检测器与假视频源，真实 SQLite、EventPipeline、VlmReviewer 与 PatrolRuntime。

**Validates: Requirements 5.5, 5.7, 12.4**
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chimera_patrol.config import parse_config
from chimera_patrol.runtime import PatrolRuntime
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult, ReviewRequest, SopCheckRequest
from chimera_patrol.vlm.normalize import VLMVerdict

FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]
PERSON_BOX = (100.0, 50.0, 160.0, 230.0)


class _Detector:
    def detect_persons(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        return [(PERSON_BOX, 0.9)]

    def detect_fire_smoke(self, frame: np.ndarray) -> list[Any]:
        return []

    def detect_helmets(self, frame: np.ndarray) -> list[Any]:
        return []


class _Capture:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self._frame = np.full((240, 320, 3), 90, dtype=np.uint8)

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self) -> tuple[bool, Any]:
        return (True, self._frame.copy()) if self._opened else (False, None)

    def get(self, prop_id: int) -> float:
        return 25.0 if prop_id == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        return None


def _capture_factory(source: str) -> _Capture:
    return _Capture(opened=not source.startswith("bad"))


class _SlowAdapter:
    """复核耗时远超 timeout_s，模拟慢 VLM。"""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        await asyncio.sleep(self._delay_s)
        raise AssertionError("slow adapter should have been cancelled by timeout")

    async def check_sop(self, req: SopCheckRequest) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    async def check_helmet_crop(self, req: HelmetCropRequest) -> HelmetCropResult:
        return HelmetCropResult(verdict="uncertain", confidence=0.0)


class _RaisingAdapter(_SlowAdapter):
    """每次复核都抛出普通异常（适配层故障）。"""

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        raise RuntimeError("boom")


def _config(evidence_dir: Path, cameras: list[dict[str, Any]], *, review_timeout_s: float = 20.0) -> Any:
    return parse_config(
        {
            "evidence": {"base_dir": str(evidence_dir), "ring_buffer_s": 20, "clip_pre_s": 0, "clip_post_s": 60},
            "video": {"default_fps": 25, "max_cameras": 4},
            "detector": {"frame_skip_n": 1},
            "rules": {"cooldown_s": 30},
            "vlm": {"provider": "mock", "review_enabled": True, "timeout_s": review_timeout_s},
            "cameras": cameras,
        },
        env={},
    )


def _camera(camera_id: str) -> dict[str, Any]:
    return {
        "id": camera_id,
        "name": camera_id,
        "rois": [{"id": "zone", "type": "intrusion", "polygon": FULL_FRAME}],
    }


def _run_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _wait_for(predicate: Any, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _events_by_camera(repo: EventRepository, camera_id: str) -> list[dict[str, Any]]:
    items, _ = repo.list_events(camera_id=camera_id)
    return items


def _status_log(db_path: Path, event_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM event_status_log WHERE event_id = ? ORDER BY id", (event_id,)
    ).fetchall()]
    conn.close()
    return rows


def test_slow_vlm_does_not_block_detection_and_times_out_as_review_failed(tmp_path: Path) -> None:
    db_path = tmp_path / "patrol.db"
    repo = EventRepository(db_path)
    loop, thread = _run_loop()
    cfg = _config(
        tmp_path / "evidence",
        [_camera("cam_a"), _camera("cam_b")],
        review_timeout_s=0.3,
    )
    runtime = PatrolRuntime(
        cfg,
        repo,
        _Detector(),
        loop,
        candidates=["cam_a.mp4", "cam_b.mp4"],
        capture_factory=_capture_factory,
        adapter=_SlowAdapter(delay_s=10.0),
    )
    try:
        started = time.monotonic()
        runtime.start()

        # 两路都要在慢 VLM 仍未返回时产生事件，说明检测主链路未被复核阻塞。
        assert _wait_for(lambda: _events_by_camera(repo, "cam_a") and _events_by_camera(repo, "cam_b")), (
            "detection stalled while VLM is slow"
        )
        assert time.monotonic() - started < 9.0

        # 超时后两个事件都应进入 review_failed，错误原因为 timeout。
        assert _wait_for(
            lambda: all(
                e["status"] == "review_failed" for cam in ("cam_a", "cam_b") for e in _events_by_camera(repo, cam)
            )
        ), "events did not reach review_failed after VLM timeout"
        for cam in ("cam_a", "cam_b"):
            assert _events_by_camera(repo, cam)[0]["vlm_error"] == "timeout"
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        repo.close()


def test_adapter_exception_fails_only_review_and_keeps_all_cameras_running(tmp_path: Path) -> None:
    db_path = tmp_path / "patrol.db"
    repo = EventRepository(db_path)
    loop, thread = _run_loop()
    runtime = PatrolRuntime(
        _config(tmp_path / "evidence", [_camera("cam_a"), _camera("cam_b")]),
        repo,
        _Detector(),
        loop,
        candidates=["cam_a.mp4", "cam_b.mp4"],
        capture_factory=_capture_factory,
        adapter=_RaisingAdapter(delay_s=0.0),
    )
    try:
        runtime.start()
        assert _wait_for(
            lambda: all(
                any(e["status"] == "review_failed" for e in _events_by_camera(repo, cam))
                for cam in ("cam_a", "cam_b")
            )
        ), "both cameras should keep producing reviewed events despite adapter errors"
        for cam in ("cam_a", "cam_b"):
            failed = [e for e in _events_by_camera(repo, cam) if e["status"] == "review_failed"]
            assert failed[0]["vlm_error"].startswith("vlm_error: RuntimeError: boom")
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        repo.close()


def test_startup_recovery_marks_leftover_pending_events_as_review_failed(tmp_path: Path) -> None:
    db_path = tmp_path / "patrol.db"
    repo = EventRepository(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
        "VALUES ('cam_pre', 'Pre', 'file', 'pre.mp4', 0)"
    )
    conn.commit()
    conn.close()
    repo.insert_event(
        NewEvent(
            id="evt_leftover000001",
            camera_id="cam_pre",
            event_type="intrusion",
            detection_source="yolo",
            start_ms=1_000,
            end_ms=1_000,
            keyframe_annotated_path="cam_pre/evt_leftover000001/keyframe_annotated.jpg",
            keyframe_raw_path="cam_pre/evt_leftover000001/keyframe_raw.jpg",
            created_ms=1_000,
            cv_confidence=0.8,
            track_id=1,
            roi_id="zone",
            roi_type="intrusion",
            cv_detail={"label": "person"},
            trajectory=[],
        )
    )
    loop, thread = _run_loop()
    runtime = PatrolRuntime(
        _config(tmp_path / "evidence", [_camera("cam_a")], review_timeout_s=0.2),
        repo,
        _Detector(),
        loop,
        candidates=["cam_a.mp4"],
        capture_factory=_capture_factory,
        adapter=_SlowAdapter(delay_s=10.0),
    )
    try:
        runtime.start()
        assert runtime.recovered_count == 1
        event = repo.get_event("evt_leftover000001")
        assert event is not None
        assert event["status"] == "review_failed"
        assert event["vlm_error"] == "service_restart"
        last = _status_log(db_path, "evt_leftover000001")[-1]
        assert last["to_status"] == "review_failed"
        assert last["actor"] == "startup_recovery"
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        repo.close()
