"""DetectWorker：每路 Camera 一个检测线程（任务 3.3，design「线程与异步模型」）。

流程：从 DetectQueue 取帧（已按抽帧间隔筛选）→ Detector 检测 person、烟火、安全帽 →
PersonTracker 跟踪 person → RuleEngine.process 产出候选事件 → 通过 ``submit`` 交给事件循环。

- ``submit`` 由调用方提供（runtime 中是 ``run_coroutine_threadsafe`` 的封装），失败只记录日志。
- ROI 在第一帧到达时按帧尺寸从归一化坐标换算为像素坐标，之后保持不变。
- 视频循环重启（假设 A4）通过 ``request_reset`` 请求；线程在下一帧前清空跟踪器与规则状态。
- 单帧处理异常只记录日志并丢弃该帧，线程继续运行，不影响其他路。
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

import cv2

from chimera_patrol.config import RoiConfig, RulesSection
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.ingest.capture import DetectItem, DetectQueue
from chimera_patrol.rules.engine import RuleEngine
from chimera_patrol.rules.helmet_fallback import FallbackRequest, crop_head_region
from chimera_patrol.rules.smoke_fire import SmokeFireDetection
from chimera_patrol.rules.types import (
    CandidateEvent,
    FrameResult,
    OverlayItem,
    HelmetDetection,
    PersonDetection,
    RoiSpec,
)
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult

log = logging.getLogger("chimera_patrol.detect.worker")

Submit = Callable[[CandidateEvent], None]
# 把头部裁剪请求投递到事件循环，返回线程安全的 Future；结果为 None 表示兜底失败。
FallbackSubmit = Callable[[HelmetCropRequest], "concurrent.futures.Future[HelmetCropResult | None]"]
_PendingFallback = tuple[FallbackRequest, "concurrent.futures.Future[HelmetCropResult | None]"]


def _overlay_items(result: FrameResult) -> list[OverlayItem]:
    """把一帧的检测结果转为可绘制的标注框（供 MJPEG 叠加）。"""
    items: list[OverlayItem] = [
        OverlayItem(box=person.box, label="person", conf=person.conf) for person in result.persons
    ]
    items.extend(
        OverlayItem(box=det.box, label=det.label, conf=det.conf) for det in result.smoke_fire
    )
    items.extend(
        OverlayItem(box=det.box, label=det.kind, conf=det.conf) for det in result.helmets
    )
    return items


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
        fallback_submit: FallbackSubmit | None = None,
        max_pending_fallback: int = 1,
        fallback_jpeg_quality: int = 85,
        frame_slot: Any = None,
    ) -> None:
        """
        Args:
            frame_slot: 可选的 FrameSlot。每处理一帧后把检测框写入其 ``overlay``，供 MJPEG 叠加显示。
            fallback_submit: 安全帽 VLM 兜底的投递函数。为 None 时不做兜底（开关关闭或复核关闭）。
            max_pending_fallback: 本路同时在途的兜底请求上限，取自 vlm.max_concurrency。
                达到上限时跳过新的兜底请求，该 track 的请求在间隔后再次尝试。
            fallback_jpeg_quality: 头部裁剪图的 JPEG 质量（vlm.image_jpeg_quality）。
        """
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
        self._fallback_submit = fallback_submit
        self._max_pending_fallback = max_pending_fallback
        self._fallback_jpeg_quality = fallback_jpeg_quality
        self._frame_slot = frame_slot
        self._pending: list[_PendingFallback] = []
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
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=self._poll_s)
                except queue.Empty:
                    continue
                try:
                    self._handle(item)
                except Exception:
                    log.exception("检测帧处理失败 camera=%s frame_index=%s", self._camera_id, item.frame_index)
        finally:
            self._cancel_pending()

    def _handle(self, item: DetectItem) -> None:
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._tracker.reset()
            self._cancel_pending()
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
        if self._frame_slot is not None:
            self._frame_slot.overlay = _overlay_items(result)
        self._drain_fallback()
        for candidate in self._engine.process(result):
            self._emit(candidate)
        self._dispatch_fallback(result, frame)

    # ------------------------------------------------------------------ VLM 兜底（3.7）

    def _emit(self, candidate: CandidateEvent) -> None:
        try:
            self._submit(candidate)
        except Exception:
            log.exception("候选事件投递失败 camera=%s type=%s", self._camera_id, candidate.event_type)

    def _dispatch_fallback(self, result: FrameResult, frame: Any) -> None:
        """对作用域内未戴帽的 person 裁剪头部并投递到事件循环。不等待结果。

        裁剪与 JPEG 编码在本线程完成（小图，开销低）；VLM 调用在事件循环中执行。
        在途请求数达到上限时直接返回，不消耗请求间隔，下一帧会重新尝试。
        """
        if self._fallback_submit is None or self._engine is None:
            return
        if len(self._pending) >= self._max_pending_fallback:
            return
        helmet_rules = self._rules.helmet
        for request in self._engine.collect_fallback_requests(result):
            crop = crop_head_region(
                frame,
                request.box,
                helmet_rules.head_region_ratio,
                helmet_rules.vlm_fallback.crop_padding_ratio,
            )
            if crop is None:
                continue
            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self._fallback_jpeg_quality])
            if not ok:
                log.warning("头部裁剪编码失败 camera=%s track=%s", self._camera_id, request.track_id)
                continue
            payload = HelmetCropRequest(
                track_id=str(request.track_id),
                camera_id=self._camera_id,
                image_jpeg=buffer.tobytes(),
            )
            try:
                future = self._fallback_submit(payload)
            except Exception:
                log.exception("兜底请求投递失败 camera=%s track=%s", self._camera_id, request.track_id)
                continue
            self._pending.append((request, future))
            if len(self._pending) >= self._max_pending_fallback:
                break

    def _drain_fallback(self) -> None:
        """取出已完成的兜底结果，在本线程中调用 apply_fallback_result 生成候选事件。"""
        if not self._pending or self._engine is None:
            return
        still_running: list[_PendingFallback] = []
        for request, future in self._pending:
            if not future.done():
                still_running.append((request, future))
                continue
            if future.cancelled():
                continue
            error = future.exception()
            if error is not None:
                log.error("兜底结果异常 camera=%s track=%s error=%s", self._camera_id, request.track_id, error)
                continue
            result = future.result()
            if result is None:
                continue
            candidate = self._engine.apply_fallback_result(request, result.verdict, result.confidence)
            if candidate is not None:
                self._emit(candidate)
        self._pending = still_running

    def _cancel_pending(self) -> None:
        for _request, future in self._pending:
            future.cancel()
        self._pending = []