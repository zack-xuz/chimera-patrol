"""运行时装配（任务 6.2、10.7、10.8、11.2）：按配置启动复核、兜底与每路 Camera 的线程。

启动顺序（``PatrolRuntime.start``）：

1. 启动恢复（10.7）：遗留的 pending_review 事件迁移为 review_failed（service_restart）。
   在任何新事件产生之前完成，因此不会误伤本次运行中的复核。
2. 发现 Camera，登记 cameras 表，为每路创建 RingBuffer。
3. 装配 VLM：VlmReviewer（10.5）经 build_on_review（10.8）接入 EventPipeline 的 on_review。
   复核关闭时同样经过 VlmReviewer，由它把事件迁移为 needs_human。
4. 安全帽兜底（11.2）：开关与复核都开启时，每路 DetectWorker 持有投递函数，
   VLM 调用在事件循环中执行，结果经 Future 回到工作线程。
5. 启动每路 CaptureThread 与 DetectWorker。

线程模型：

- DetectWorker 通过 ``run_coroutine_threadsafe(pipeline.submit(candidate), loop)`` 投递候选事件。
- 兜底请求通过 ``run_coroutine_threadsafe(checker.check(req), loop)`` 投递，
  返回的 Future 由工作线程轮询，规则状态只在工作线程修改。
- 单路异常只影响该路（各线程内部捕获异常）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from chimera_patrol.config import AppConfig
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.detect.worker import DetectWorker
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.ingest.camera import Camera, mask_source
from chimera_patrol.ingest.capture import CaptureThread, DetectQueue, FrameSlot
from chimera_patrol.ingest.discover import discover_cameras
from chimera_patrol.logging_setup import register_secret
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.review.helmet_check import HelmetCropChecker
from chimera_patrol.review.recovery import recover_pending_reviews
from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.review.wiring import build_on_review
from chimera_patrol.rules.types import CandidateEvent
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.vlm.factory import build_adapter
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult, VLMAdapter

log = logging.getLogger("chimera_patrol.runtime")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _log_future(future: Future[Any]) -> None:
    try:
        future.result()
    except Exception:
        log.exception("候选事件落库失败")


class PatrolRuntime:
    """全部组件装配与生命周期管理。

    Args:
        config: 已校验的 AppConfig。
        repository: 事件存储。
        detector: 具备 detect_persons / detect_fire_smoke / detect_helmets 的检测器。
        loop: 运行 EventPipeline 的事件循环（需在另一线程中运行）。
        candidates: 可选的候选视频文件列表，缺省扫描 ``video.source_dir``。
        capture_factory: 可选的 VideoCapture 工厂，测试时注入假源。
        adapter: 可选的 VLM_Adapter。缺省时：复核开启则按配置构造，复核关闭则不构造。
    """

    def __init__(
        self,
        config: AppConfig,
        repository: EventRepository,
        detector: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        candidates: Any = None,
        capture_factory: Callable[[str], Any] | None = None,
        adapter: VLMAdapter | None = None,
    ) -> None:
        self._config = config
        self._repo = repository
        self._detector = detector
        self._loop = loop
        self._candidates = candidates
        self._capture_factory = capture_factory
        self._adapter_override = adapter
        self._cameras: list[Camera] = []
        self._captures: list[CaptureThread] = []
        self._workers: list[DetectWorker] = []
        self._pipeline: EventPipeline | None = None
        self._checker: HelmetCropChecker | None = None
        self._recovered_count = 0

    @property
    def cameras(self) -> Sequence[Camera]:
        return tuple(self._cameras)

    @property
    def recovered_count(self) -> int:
        """启动时被恢复为 review_failed 的遗留事件数。"""
        return self._recovered_count

    def start(self) -> None:
        """恢复遗留事件，装配复核与兜底，启动所有线程。"""
        cfg = self._config
        state = EventStateMachine(self._repo)
        self._recovered_count = recover_pending_reviews(self._repo, state, now_ms=_now_ms())
        if self._recovered_count:
            log.info("startup recovery moved %d pending events to review_failed", self._recovered_count)

        self._cameras = discover_cameras(cfg.video, cfg.cameras, self._candidates)
        now = _now_ms()
        ring_buffers: dict[str, RingBuffer] = {}

        for cam in self._cameras:
            self._repo.upsert_camera(
                cam.id,
                name=cam.name,
                source_type=cam.source_type,
                source=mask_source(cam.source),
                status="offline",
                now_ms=now,
            )
            ring_buffers[cam.id] = RingBuffer(window_s=cfg.evidence.ring_buffer_s, max_fps=cfg.evidence.ring_fps)

        # 复核与兜底共用同一并发上限（vlm.max_concurrency）。
        semaphore = asyncio.Semaphore(cfg.vlm.max_concurrency)
        adapter = self._resolve_adapter()
        reviewer = VlmReviewer(
            adapter,
            state,
            self._repo,
            timeout_s=cfg.vlm.timeout_s,
            max_concurrency=cfg.vlm.max_concurrency,
            description_max_chars=cfg.vlm.description_max_chars,
            auto_decision_threshold=cfg.vlm.auto_decision_threshold,
            review_enabled=cfg.vlm.review_enabled,
            semaphore=semaphore,
        )
        on_review = build_on_review(
            repository=self._repo,
            reviewer=reviewer,
            base_dir=Path(cfg.evidence.base_dir),
            camera_names={cam.id: cam.name for cam in self._cameras},
        )

        self._pipeline = EventPipeline(
            repository=self._repo,
            ring_buffers=ring_buffers,
            evidence=cfg.evidence,
            on_review=on_review,
        )

        fallback_enabled = (
            adapter is not None and cfg.vlm.review_enabled and cfg.rules.helmet.vlm_fallback.enabled
        )
        self._checker = (
            HelmetCropChecker(adapter, timeout_s=cfg.vlm.timeout_s, semaphore=semaphore)
            if fallback_enabled and adapter is not None
            else None
        )

        for cam in self._cameras:
            queue = DetectQueue(maxsize=1)
            tracker = PersonTracker(
                frame_rate=cfg.video.default_fps,
                lost_buffer_s=cfg.detector.tracker.lost_buffer_s,
            )
            worker = DetectWorker(
                camera_id=cam.id,
                detect_queue=queue,
                detector=self._detector,
                tracker=tracker,
                rules=cfg.rules,
                rois=cam.rois,
                submit=self._submit,
                vlm_review_enabled=cfg.vlm.review_enabled,
                fallback_submit=self._submit_fallback if self._checker is not None else None,
                max_pending_fallback=cfg.vlm.max_concurrency,
                fallback_jpeg_quality=cfg.vlm.image_jpeg_quality,
            )
            kwargs: dict[str, Any] = {}
            if self._capture_factory is not None:
                kwargs["capture_factory"] = self._capture_factory
            capture = CaptureThread(
                cam,
                cfg.video,
                cfg.detector.frame_skip_n,
                FrameSlot(),
                queue,
                self._on_status_change,
                ring_buffer=ring_buffers[cam.id],
                ring_fps=cfg.evidence.ring_fps,
                ring_jpeg_quality=cfg.evidence.ring_jpeg_quality,
                on_loop_restart=lambda camera_id, ts, w=worker: w.request_reset(),
                **kwargs,
            )
            self._workers.append(worker)
            self._captures.append(capture)

        for worker in self._workers:
            worker.start()
        for capture in self._captures:
            capture.start()
        log.info(
            "runtime started cameras=%d review_enabled=%s helmet_fallback=%s",
            len(self._cameras),
            cfg.vlm.review_enabled,
            self._checker is not None,
        )

    def stop(self, timeout_s: float = 10.0) -> None:
        """停止所有采集与检测线程，并关闭流水线（取消后台片段任务）。

        正在进行中的 VLM 复核不会被取消。它们留下的 pending_review 事件由下次启动的恢复逻辑处理。
        """
        for capture in self._captures:
            capture.stop()
        for worker in self._workers:
            worker.stop()
        for capture in self._captures:
            capture.join(timeout_s)
        for worker in self._workers:
            worker.join(timeout_s)
        if self._pipeline is not None:
            fut = asyncio.run_coroutine_threadsafe(self._pipeline.close(), self._loop)
            fut.result(timeout=timeout_s)
            self._pipeline = None
        self._checker = None
        self._captures.clear()
        self._workers.clear()

    # 内部方法

    def _resolve_adapter(self) -> VLMAdapter | None:
        """返回注入的适配器；否则复核开启时按配置构造，复核关闭时返回 None。

        API Key 只从环境变量读取，并登记为日志脱敏内容（12.6）。
        """
        if self._adapter_override is not None:
            return self._adapter_override
        vlm = self._config.vlm
        if not vlm.review_enabled:
            return None
        api_key = os.environ.get(vlm.openai_compatible.api_key_env) or None
        register_secret(api_key)
        return build_adapter(vlm, api_key)

    def _submit(self, candidate: CandidateEvent) -> None:
        if self._pipeline is None:
            raise RuntimeError("runtime is not started")
        future = asyncio.run_coroutine_threadsafe(self._pipeline.submit(candidate), self._loop)
        future.add_done_callback(_log_future)

    def _submit_fallback(self, request: HelmetCropRequest) -> Future[HelmetCropResult | None]:
        if self._checker is None:
            raise RuntimeError("helmet fallback is not enabled")
        return asyncio.run_coroutine_threadsafe(self._checker.check(request), self._loop)

    def _on_status_change(self, camera_id: str, status: str, ts_ms: int) -> None:
        try:
            self._repo.set_camera_status(camera_id, status, ts_ms)
        except Exception:
            log.exception("更新 Camera 状态失败 camera=%s", camera_id)
