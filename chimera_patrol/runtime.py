"""运行时装配（任务 6.2、10.7、10.8、11.2、15.8）：按配置启动复核、兜底、SOP 与每路 Camera 的线程。

启动顺序（``PatrolRuntime.start``）：

1. 启动恢复（10.7、15.7）：遗留的 pending_review 事件迁移为 review_failed（service_restart），
   遗留的 running SOP 检查置为 check_failed_retry（service_restart）。
   在任何新事件产生之前完成。
2. 发现 Camera，登记 cameras 表，为每路创建 RingBuffer 与 FrameSlot。
3. 装配 VLM：VlmReviewer 经 build_on_review（10.8）接入 EventPipeline 的 on_review。
   复核关闭时同样经过 VlmReviewer，由它把事件迁移为 needs_human。
   适配器无论复核开关如何都尝试构造，SOP 检查需要它；缺密钥且复核关闭时适配器为 None。
4. 装配 SopChecker（15.8），加载 sops.yaml。
5. 安全帽兜底（11.2）：开关与复核都开启时，每路 DetectWorker 持有投递函数，
   VLM 调用在事件循环中执行，结果经 Future 回到工作线程。
6. 启动每路 CaptureThread 与 DetectWorker。

推送：传入 ``notifier`` 时，事件创建、片段完成、复核完成、摄像头状态变化、SOP 检查状态变化
都会通过它推送到事件循环（``EventNotifier`` 协议）。未传入时不推送。

线程模型：

- DetectWorker 通过 ``run_coroutine_threadsafe(pipeline.submit(candidate), loop)`` 投递候选事件。
- 兜底请求通过 ``run_coroutine_threadsafe(checker.check(request), loop)`` 投递，
  返回的 Future 由工作线程轮询，规则状态只在工作线程修改。
- 单路异常只影响该路（各线程内部捕获异常）。
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import wait as futures_wait
from pathlib import Path
from typing import Any, Protocol

from chimera_patrol.config import AppConfig
from chimera_patrol.control import CameraControl
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.detect.worker import DetectWorker
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.ingest.camera import Camera, mask_source
from chimera_patrol.ingest.capture import CaptureThread, DetectQueue, FrameSlot
from chimera_patrol.ingest.discover import discover_cameras
from chimera_patrol.logging_setup import register_secret
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.review.helmet_check import HelmetCropChecker
from chimera_patrol.review.recovery import recover_pending_reviews, recover_running_sop_runs
from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.review.wiring import build_on_review
from chimera_patrol.rules.types import CandidateEvent
from chimera_patrol.sop.checker import SopChecker
from chimera_patrol.sop.loader import load_sops
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.sop_repo import SopRepository
from chimera_patrol.vlm.factory import build_adapter
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult, VLMAdapter

log = logging.getLogger("chimera_patrol.runtime")


class EventNotifier(Protocol):
    """事件循环中的推送接口。实现由 API 层提供（WsHub 之上的薄封装）。"""

    async def on_created(self, summary: dict[str, Any]) -> None: ...

    async def on_clip_done(self, event_id: str, status: str) -> None: ...

    async def on_event_updated(self, event_id: str) -> None: ...

    async def on_camera_status(self, camera_id: str, status: str, ts_ms: int) -> None: ...

    async def on_sop_run(self, run_id: str) -> None: ...


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
        loop: 运行 EventPipeline 的事件循环（需在另一线程中运行，或由 uvicorn 持有）。
        candidates: 可选的候选视频文件列表，缺省扫描 ``video.source_dir``。
        capture_factory: 可选的 VideoCapture 工厂，测试时注入假源。
        adapter: 可选的 VLM_Adapter。缺省时按配置构造。
        notifier: 可选的推送实现（EventNotifier）。
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
        notifier: EventNotifier | None = None,
        control: CameraControl | None = None,
    ) -> None:
        self._config = config
        self._control = control if control is not None else CameraControl()
        self._repo = repository
        self._sop_repo = SopRepository(repository)
        self._detector = detector
        self._loop = loop
        self._candidates = candidates
        self._capture_factory = capture_factory
        self._adapter_override = adapter
        self._notifier = notifier
        self._cameras: list[Camera] = []
        self._captures: list[CaptureThread] = []
        self._workers: list[DetectWorker] = []
        self._frame_slots: dict[str, FrameSlot] = {}
        self._pipeline: EventPipeline | None = None
        self._checker: HelmetCropChecker | None = None
        self._sop: SopChecker | None = None
        self._recovered_count = 0
        self._recovered_sop_count = 0
        # 已投递到事件循环、尚未完成的候选事件。关闭时需等待它们完成后才能关闭流水线。
        self._submit_lock = threading.Lock()
        self._pending_submits: set[Future[Any]] = set()

    @property
    def cameras(self) -> Sequence[Camera]:
        return tuple(self._cameras)

    @property
    def control(self) -> CameraControl:
        """每路检测的暂停登记表（内存，不落库）。"""
        return self._control

    @property
    def frame_slots(self) -> dict[str, FrameSlot]:
        """每路 Camera 的 FrameSlot，供 MJPEG 读取最新帧与检测框。"""
        return dict(self._frame_slots)

    @property
    def sop_checker(self) -> SopChecker | None:
        return self._sop

    @property
    def recovered_count(self) -> int:
        """启动时被恢复为 review_failed 的遗留事件数。"""
        return self._recovered_count

    @property
    def recovered_sop_count(self) -> int:
        """启动时被恢复为 check_failed_retry 的遗留 SOP 检查数。"""
        return self._recovered_sop_count

    def start(self) -> None:
        """恢复遗留事件，装配复核、兜底与 SOP，启动所有线程。"""
        cfg = self._config
        now = _now_ms()
        state = EventStateMachine(self._repo)
        self._recovered_count = recover_pending_reviews(self._repo, state, now_ms=now)
        if self._recovered_count:
            log.info("startup recovery moved %d pending events to review_failed", self._recovered_count)
        self._recovered_sop_count = recover_running_sop_runs(self._sop_repo, now_ms=now)
        if self._recovered_sop_count:
            log.info("startup recovery moved %d running SOP checks to check_failed_retry", self._recovered_sop_count)

        self._cameras = discover_cameras(cfg.video, cfg.cameras, self._candidates)
        now = _now_ms()
        ring_buffers: dict[str, RingBuffer] = {}
        self._frame_slots = {}

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
            self._frame_slots[cam.id] = FrameSlot()

        # 复核、兜底与 SOP 共用同一并发上限（vlm.max_concurrency）。
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
        review_hook = build_on_review(
            repository=self._repo,
            reviewer=reviewer,
            base_dir=Path(cfg.evidence.base_dir),
            camera_names={cam.id: cam.name for cam in self._cameras},
        )

        async def on_review(event_id: str) -> None:
            await review_hook(event_id)
            if self._notifier is not None:
                await self._notifier.on_event_updated(event_id)

        self._pipeline = EventPipeline(
            repository=self._repo,
            ring_buffers=ring_buffers,
            evidence=cfg.evidence,
            on_created=self._notifier.on_created if self._notifier else None,
            on_clip_done=self._notifier.on_clip_done if self._notifier else None,
            on_review=on_review,
        )

        self._sop = SopChecker(
            definitions=load_sops(cfg.sop.file),
            repository=self._repo,
            sop_repo=self._sop_repo,
            pipeline=self._pipeline,
            state=state,
            adapter=adapter,
            ring_buffers=ring_buffers,
            sop_cfg=cfg.sop,
            evidence_cfg=cfg.evidence,
            timeout_s=cfg.vlm.timeout_s,
            semaphore=semaphore,
            notifier=self._notifier,
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
                frame_slot=self._frame_slots[cam.id],
                pause_check=functools.partial(self._control.is_paused, cam.id),
            )
            kwargs: dict[str, Any] = {}
            if self._capture_factory is not None:
                kwargs["capture_factory"] = self._capture_factory
            capture = CaptureThread(
                cam,
                cfg.video,
                cfg.detector.frame_skip_n,
                self._frame_slots[cam.id],
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
            "runtime started cameras=%d review_enabled=%s helmet_fallback=%s sop_checks=%d",
            len(self._cameras),
            cfg.vlm.review_enabled,
            self._checker is not None,
            len(self._sop.definitions),
        )

    def stop(self, timeout_s: float = 10.0) -> None:
        """停止所有采集与检测线程，并关闭流水线（取消后台片段任务）。

        从非事件循环线程调用。事件循环内请使用 ``ashutdown``，否则会死锁。
        正在进行中的 VLM 复核不会被取消。它们留下的 pending_review 事件由下次启动的恢复逻辑处理。
        """
        self._stop_threads(timeout_s)
        if self._pipeline is not None:
            fut = asyncio.run_coroutine_threadsafe(self._close_async_parts(), self._loop)
            fut.result(timeout=timeout_s)
        self._pipeline = None

    async def ashutdown(self, timeout_s: float = 60.0) -> None:
        """在事件循环内关闭。顺序：先停采集与检测线程并等待其退出，再关闭流水线与 SOP 任务。

        检测线程在退出前仍可能向事件循环投递候选事件，因此流水线必须在线程退出之后关闭。
        """
        await asyncio.to_thread(self._stop_threads, timeout_s)
        await self._close_async_parts()
        self._pipeline = None

    # 内部方法

    def _stop_threads(self, timeout_s: float) -> None:
        """请求所有采集与检测线程退出并等待；再等待已投递的候选事件处理完成。

        超时仍未退出的线程、仍未完成的候选事件都记录为错误。仅在两者都结束后才允许关闭流水线。
        """
        threads: list[threading.Thread] = [*self._captures, *self._workers]
        for capture in self._captures:
            capture.stop()
        for worker in self._workers:
            worker.stop()
        still_alive: list[str] = []
        for thread in threads:
            thread.join(timeout_s)
            if thread.is_alive():
                still_alive.append(thread.name)
        if still_alive:
            log.error("线程未在 %.0f 秒内退出，仍在运行: %s", timeout_s, ", ".join(still_alive))

        with self._submit_lock:
            pending = list(self._pending_submits)
        if pending:
            log.info("等待 %d 个已投递的候选事件处理完成", len(pending))
            _, not_done = futures_wait(pending, timeout=timeout_s)
            if not_done:
                log.error("候选事件未在 %.0f 秒内处理完成，数量 %d", timeout_s, len(not_done))
        self._captures.clear()
        self._workers.clear()
        self._checker = None

    async def _close_async_parts(self) -> None:
        if self._sop is not None:
            await self._sop.close()
            self._sop = None
        if self._pipeline is not None:
            await self._pipeline.close()

    def _resolve_adapter(self) -> VLMAdapter | None:
        """返回注入的适配器；否则按配置构造。

        API Key 只从环境变量读取，并登记为日志脱敏内容（12.6）。缺密钥时：
        复核开启则配置校验已拒绝启动，此处不会出现；复核关闭则返回 None，
        SOP 检查会记为检查失败待重试。
        """
        if self._adapter_override is not None:
            return self._adapter_override
        vlm = self._config.vlm
        api_key = os.environ.get(vlm.openai_compatible.api_key_env) or None
        register_secret(api_key)
        try:
            return build_adapter(vlm, api_key)
        except ValueError as exc:
            if vlm.review_enabled:
                raise
            log.warning("VLM 适配器未配置，SOP 检查将记为检查失败待重试: %s", exc)
            return None

    def _submit(self, candidate: CandidateEvent) -> None:
        if self._pipeline is None:
            raise RuntimeError("runtime is not started")
        future = asyncio.run_coroutine_threadsafe(self._pipeline.submit(candidate), self._loop)
        with self._submit_lock:
            self._pending_submits.add(future)
        future.add_done_callback(self._submit_done)

    def _submit_done(self, future: Future[Any]) -> None:
        with self._submit_lock:
            self._pending_submits.discard(future)
        _log_future(future)

    def _submit_fallback(self, request: HelmetCropRequest) -> Future[HelmetCropResult | None]:
        if self._checker is None:
            raise RuntimeError("helmet fallback is not enabled")
        return asyncio.run_coroutine_threadsafe(self._checker.check(request), self._loop)

    def _on_status_change(self, camera_id: str, status: str, ts_ms: int) -> None:
        try:
            self._repo.set_camera_status(camera_id, status, ts_ms)
        except Exception:
            log.exception("更新 Camera 状态失败 camera=%s", camera_id)
            return
        if self._notifier is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._notifier.on_camera_status(camera_id, status, ts_ms), self._loop
            )
            future.add_done_callback(lambda f: _log_future(f))
