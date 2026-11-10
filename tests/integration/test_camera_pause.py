"""每路检测暂停（运行时层面）：暂停后该路不再产生新事件，另一路照常，恢复后继续产生。

假检测器通过帧内编码的计数器决定烟火框位置：每 3 个已处理帧换一个位置，
使每 3 帧产生一次 smoke_fire 事件（冷却为 0）。真实 YOLO 不参与。
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chimera_patrol.config import parse_config
from chimera_patrol.runtime import PatrolRuntime
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.vlm.mock import MockAdapter

FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]


class CountingCapture:
    """假视频源：每帧把计数器写入左上角像素，供检测器读取。"""

    def __init__(self) -> None:
        self._n = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, Any]:
        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        frame[0, 0, 0] = self._n % 256
        self._n += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        return 25.0 if prop_id == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        return None


class SmokeEveryThreeFrames:
    """检测器：框位置由帧计数器决定，每 3 帧换一个位置。"""

    def detect_persons(self, frame: np.ndarray) -> list[Any]:
        return []

    def detect_helmets(self, frame: np.ndarray) -> list[Any]:
        return []

    def detect_fire_smoke(self, frame: np.ndarray) -> list[tuple[str, tuple[float, float, float, float], float]]:
        group = int(frame[0, 0, 0]) // 3
        x0 = float((group * 40) % 120)
        return [("smoke", (x0, 10.0, x0 + 30.0, 40.0), 0.9)]


def _config(evidence_dir: Path) -> Any:
    return parse_config(
        {
            "evidence": {"base_dir": str(evidence_dir), "ring_buffer_s": 20, "clip_pre_s": 0, "clip_post_s": 60},
            "video": {"default_fps": 25, "max_cameras": 4},
            "detector": {"frame_skip_n": 1},
            "rules": {"cooldown_s": 0},
            "vlm": {"provider": "mock"},
            "cameras": [
                {"id": "cam_a", "name": "A", "rois": [{"id": "z", "type": "intrusion", "polygon": FULL_FRAME}]},
                {"id": "cam_b", "name": "B", "rois": [{"id": "z", "type": "intrusion", "polygon": FULL_FRAME}]},
            ],
        },
        env={"VLM_API_KEY": "unused-mock-provider"},
    )


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _count(repo: EventRepository, camera_id: str) -> int:
    return repo.list_events(camera_id=camera_id, event_type="smoke_fire", limit=1)[1]


def test_pause_stops_one_camera_and_resume_restarts_it(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "evidence")
    repo = EventRepository(tmp_path / "chimera.db")
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    runtime = PatrolRuntime(
        cfg,
        repo,
        SmokeEveryThreeFrames(),
        loop,
        capture_factory=lambda source: CountingCapture(),
        adapter=MockAdapter(default_verdict="uncertain"),
        # 显式指定候选视频，避免扫描 samples/videos 接入真实样例。
        candidates=["cam_a.mp4", "cam_b.mp4"],
    )
    try:
        runtime.start()
        assert _wait_for(lambda: _count(repo, "cam_a") >= 3 and _count(repo, "cam_b") >= 3, 30), "no events"

        runtime.control.pause("cam_a")
        time.sleep(0.5)  # 允许暂停前已在处理的一帧结束
        a_paused = _count(repo, "cam_a")
        b_before = _count(repo, "cam_b")
        time.sleep(1.5)
        assert _count(repo, "cam_a") == a_paused, "paused camera kept producing events"
        assert _count(repo, "cam_b") > b_before, "other camera stopped while one was paused"

        runtime.control.resume("cam_a")
        assert _wait_for(lambda: _count(repo, "cam_a") > a_paused, 30), "camera did not resume"
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        repo.close()
