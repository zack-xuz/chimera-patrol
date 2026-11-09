"""DetectWorker：每路 Camera 一个检测线程（任务 3.3，design「线程与异步模型」）。

流程：从 DetectQueue 取帧（已按抽帧间隔筛选）→ Detector 检测 person、烟火、安全帽 →
PersonTracker 跟踪 person → RuleEngine.process 产出候选事件 → 通过 ``submit`` 交给事件循环。

- ``submit`` 由调用方提供（runtime 中是 ``run_coroutine_threadsafe`` 的封装），失败只记录日志。
- ROI 在第一帧到达时按帧尺寸从归一化坐标换算为像素坐标，之后保持不变。
- 视频循环重启（假设 A4）通过 ``request_reset`` 请求；线程在下一帧前清空跟踪器与规则状态。
- 单帧处理异常只记录日志并丢弃该帧，线程继续运行，不影响其他路。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

from chimera_patrol.config import RoiConfig, RulesSection
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.ingest.capture import DetectItem, DetectQueue
from chimera_patrol.rules.engine import RuleEngine
from chimera_patrol.rules.smoke_fire import SmokeFireDetection
from chimera_patrol.rules.types import (
    CandidateEvent,
    FrameResult,
    HelmetDetection,
    PersonDetection,
    RoiSpec,
)

log = logging.getLogger("chimera_patrol.detect.worker")

Submit = Callable[[CandidateEvent], None]


def scale_rois(rois: Sequence[RoiConfig], width: int, height: int) -> list[RoiSpec]:
    """把归一化多边形（0~1）按帧宽高换算为像素坐标。"""
    return [
        RoiSpec(
            id=roi.id,
            type=roi.type,
            polygon=[(float(x) * width, float(y) * height) for x, y in roi.polygon],
        )
        for roi in rois
    ]


class DetectWorker(threading.Thread):
    """单路 Camera 的检测与规则线程。"""

    def __init__(
        self,
        *,
        camera_id: str,
        detect_queue: DetectQueue,
        detector: Any,
        tracker: PersonTracker,
        rules: RulesSection,
        rois: Sequence[RoiConfig],
        submit: Submit,
        vlm_review_enabled: bool = True,
        poll_s: float = 0.2,
    ) -> None:
        super().__init__(name=f"detect-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._queue = detect_queue
        self._detector = detector
        self._tracker = tracker
        self._rules = rules
        self._rois = list(rois)
        self._submit = submit
        self._vlm_review_enabled = vlm_review_enabled
        self._poll_s = poll_s
        self._engine: RuleEngine | None = None
        self._reset_requested = threading.Event()
        self._stop_event = threading.Event()

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def request_reset(self) -> None:
        """请求在下一帧前清空跟踪器与规则状态（视频循环重启）。线程安全。"""
        self._reset_requested.set()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self._poll_s)
            except queue.Empty:
                continue
            try:
                self._handle(item)
            except Exception:
                log.exception("检测帧处理失败 camera=%s frame_index=%s", self._camera_id, item.frame_index)

    def _handle(self, item: DetectItem) -> None:
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._tracker.reset()
            if self._engine is not None:
                self._engine.reset()

        frame = item.image
        if self._engine is None:
            height, width = frame.shape[:2]
            self._engine = RuleEngine(
                self._camera_id,
                self._rules,
                scale_rois(self._rois, width, height),
                vlm_review_enabled=self._vlm_review_enabled,
            )

        persons_raw = self._detector.detect_persons(frame)
        tracked = self._tracker.update(persons_raw)
        result = FrameResult(
            ts_ms=item.ts_ms,
            frame_index=item.frame_index,
            persons=[PersonDetection(track_id=tid, box=box, conf=conf) for tid, box, conf in tracked],
            smoke_fire=[
                SmokeFireDetection(label=label, box=box, conf=conf)
                for label, box, conf in self._detector.detect_fire_smoke(frame)
            ],
            helmets=[
                HelmetDetection(kind=kind, box=box, conf=conf)
                for kind, box, conf in self._detector.detect_helmets(frame)
            ],
        )
        for candidate in self._engine.process(result):
            try:
                self._submit(candidate)
            except Exception:
                log.exception("候选事件投递失败 camera=%s type=%s", self._camera_id, candidate.event_type)