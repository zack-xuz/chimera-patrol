"""M1 运行时集成测试（任务 6.2、6.3）：假检测器与假视频源，真实 SQLite、流水线与证据写入。

**Validates: Requirements 1.1, 3.2, 4.1, 12.4**
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
from chimera_patrol.store.repository import EventRepository

FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]
PERSON_BOX = (100.0, 50.0, 160.0, 230.0)


class FakeDetector:
    def detect_persons(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        return [(PERSON_BOX, 0.9)]

    def detect_fire_smoke(self, frame: np.ndarray) -> list[Any]:
        return []

    def detect_helmets(self, frame: np.ndarray) -> list[Any]:
        return []


class FakeCapture:
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


def _factory(source: str) -> FakeCapture:
    return FakeCapture(opened=not source.startswith("bad"))


def _config(evidence_dir: Path) -> Any:
    return parse_config(
        {
            "evidence": {"base_dir": str(evidence_dir), "ring_buffer_s": 20, "clip_pre_s": 0, "clip_post_s": 60},
            "video": {"default_fps": 25, "max_cameras": 4},
            "detector": {"frame_skip_n": 1},
            "rules": {"cooldown_s": 30},
            "vlm": {"provider": "mock"},
            "cameras": [
                {"id": "cam_a", "name": "Camera A", "rois": [{"id": "zone", "type": "intrusion", "polygon": FULL_FRAME}]},
                {"id": "bad", "name": "Broken", "rois": [{"id": "zone", "type": "intrusion", "polygon": FULL_FRAME}]},
            ],
        },
        env={},
    )


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


def test_runtime_produces_event_on_good_camera_while_bad_camera_is_offline(tmp_path: Path) -> None:
    db_path = tmp_path / "patrol.db"
    repo = EventRepository(db_path)
    loop, thread = _run_loop()
    runtime = PatrolRuntime(
        _config(tmp_path / "evidence"),
        repo,
        FakeDetector(),
        loop,
        candidates=["cam_a.mp4", "bad.mp4"],
        capture_factory=_factory,
    )
    try:
        runtime.start()
        assert _wait_for(lambda: repo.list_events(camera_id="cam_a")[1] >= 1), "no event from cam_a"
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        repo.close()

    conn = sqlite3.connect(str(db_path))
    statuses = dict(conn.execute("SELECT id, status FROM cameras").fetchall())
    conn.close()
    assert statuses["cam_a"] == "online"
    assert statuses["bad"] == "offline"

    repo = EventRepository(db_path)
    events, total = repo.list_events()
    repo.close()
    assert total >= 1
    assert all(e["camera_id"] == "cam_a" for e in events)
    assert events[0]["event_type"] == "intrusion"
    assert events[0]["status"] == "pending_review"
    assert (tmp_path / "evidence" / events[0]["keyframe_annotated_path"]).is_file()