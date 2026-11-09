"""安全帽 VLM 兜底路径的示例测试（任务 11.3）。

假检测器不输出安全帽，因此作用域内的 person 都会触发兜底。验证：
- 兜底开启且复核开启：产生 detection_source=vlm_fallback 的 no_helmet 事件，并进入同一复核流程。
- 兜底开关关闭：适配器的 check_helmet_crop 从不被调用。
- 复核关闭：兜底被禁用，适配器从不被调用。

**Validates: Requirements 3.7, 3.8, 5.6**
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
from chimera_patrol.vlm.normalize import VLMVerdict
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult, ReviewRequest, SopCheckRequest

FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]
PERSON_BOX = (100.0, 50.0, 160.0, 230.0)


class _Detector:
    def detect_persons(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        return [(PERSON_BOX, 0.9)]

    def detect_fire_smoke(self, frame: np.ndarray) -> list[Any]:
        return []

    def detect_helmets(self, frame: np.ndarray) -> list[Any]:
        return []  # 没有安全帽检测结果，所有 person 都未关联安全帽


class _Capture:
    def __init__(self) -> None:
        self._frame = np.full((240, 320, 3), 90, dtype=np.uint8)

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, Any]:
        return True, self._frame.copy()

    def get(self, prop_id: int) -> float:
        return 25.0 if prop_id == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        return None


class _SpyAdapter:
    """记录调用次数；兜底判为未佩戴，复核判为真实事件。"""

    def __init__(self) -> None:
        self.helmet_calls = 0
        self.review_calls = 0

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        self.review_calls += 1
        return VLMVerdict(verdict="true_positive", confidence=0.9, description="未戴安全帽", suggestion="提醒佩戴")

    async def check_sop(self, req: SopCheckRequest) -> Any:  # pragma: no cover - not used here
        raise NotImplementedError

    async def check_helmet_crop(self, req: HelmetCropRequest) -> HelmetCropResult:
        self.helmet_calls += 1
        return HelmetCropResult(verdict="no_helmet", confidence=0.9)


def _config(evidence_dir: Path, *, fallback_enabled: bool, review_enabled: bool) -> Any:
    return parse_config(
        {
            "evidence": {"base_dir": str(evidence_dir), "ring_buffer_s": 20, "clip_pre_s": 0, "clip_post_s": 60},
            "video": {"default_fps": 25, "max_cameras": 4},
            "detector": {"frame_skip_n": 1},
            "rules": {
                "cooldown_s": 30,
                "helmet": {"vlm_fallback": {"enabled": fallback_enabled, "interval_s": 1}},
            },
            "vlm": {"provider": "mock", "review_enabled": review_enabled, "timeout_s": 5},
            "cameras": [
                {"id": "cam_a", "name": "Camera A", "rois": [{"id": "zone", "type": "intrusion", "polygon": FULL_FRAME}]}
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


def _fallback_events(repo: EventRepository) -> list[dict[str, Any]]:
    """只统计兜底来源的 no_helmet 事件。权重路径需要 5 秒连续违规，测试中可能也会产生。"""
    items, _ = repo.list_events(event_type="no_helmet")
    return [e for e in items if e["detection_source"] == "vlm_fallback"]


def _run(
    tmp_path: Path,
    *,
    fallback_enabled: bool,
    review_enabled: bool,
    wait_for_fallback_event: bool,
) -> tuple[EventRepository, _SpyAdapter]:
    """启动运行时，等待 intrusion 事件证明帧在流动（以及兜底事件，如需要），然后停止。"""
    repo = EventRepository(tmp_path / "patrol.db")
    loop, thread = _run_loop()
    adapter = _SpyAdapter()
    runtime = PatrolRuntime(
        _config(tmp_path / "evidence", fallback_enabled=fallback_enabled, review_enabled=review_enabled),
        repo,
        _Detector(),
        loop,
        candidates=["cam_a.mp4"],
        capture_factory=lambda _source: _Capture(),
        adapter=adapter,
    )
    try:
        runtime.start()
        # intrusion 事件在第一帧即产生，用来确认检测链路已经运行。
        assert _wait_for(lambda: repo.list_events(camera_id="cam_a")[1] >= 1), "frames did not flow"
        if wait_for_fallback_event:
            assert _wait_for(lambda: bool(_fallback_events(repo))), (
                "no vlm_fallback event produced"
            )
        else:
            # 再给兜底路径足够的帧数，确认它确实没有被调用。
            time.sleep(2.5)
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
    return repo, adapter


def test_fallback_event_goes_through_the_same_review_flow(tmp_path: Path) -> None:
    repo, adapter = _run(tmp_path, fallback_enabled=True, review_enabled=True, wait_for_fallback_event=True)
    try:
        assert adapter.helmet_calls >= 1
        event = _fallback_events(repo)[0]
        assert event["detection_source"] == "vlm_fallback"
        assert event["cv_confidence"] is not None and event["cv_confidence"] >= 0.6
        # 兜底事件与权重路径事件使用同一个复核器，因此会被复核并迁移到 confirmed。
        assert _wait_for(lambda: repo.get_event(event["id"])["status"] == "confirmed")
        assert adapter.review_calls >= 1
    finally:
        repo.close()


def test_fallback_switch_off_never_calls_helmet_adapter(tmp_path: Path) -> None:
    repo, adapter = _run(tmp_path, fallback_enabled=False, review_enabled=True, wait_for_fallback_event=False)
    try:
        assert adapter.helmet_calls == 0
        assert _fallback_events(repo) == []
    finally:
        repo.close()


def test_review_switch_off_disables_fallback_and_never_calls_adapter(tmp_path: Path) -> None:
    repo, adapter = _run(tmp_path, fallback_enabled=True, review_enabled=False, wait_for_fallback_event=False)
    try:
        assert adapter.helmet_calls == 0
        assert adapter.review_calls == 0
        assert _fallback_events(repo) == []
        # 复核关闭时，intrusion 事件直接进入 needs_human。
        assert all(e["status"] == "needs_human" for e in repo.list_events(camera_id="cam_a")[0])
    finally:
        repo.close()
