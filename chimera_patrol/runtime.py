"""M1 运行时装配（任务 6.2）：按配置启动每路 Camera 的采集线程、检测线程与事件流水线。

- 启动时登记 cameras 表（events 的外键需要），再为每路创建 RingBuffer、FrameSlot、DetectQueue。
- CaptureThread 写入 FrameSlot 与 RingBuffer，按抽帧间隔投递到 DetectQueue；循环重启时通知 DetectWorker。
- DetectWorker 产出的候选事件经 ``run_coroutine_threadsafe`` 交给事件循环中的 EventPipeline。
- 单路异常只影响该路（各线程内部捕获异常）。

M1 不装配 VLM 复核（on_review 为空操作，事件保持 pending_review），复核与恢复属于 M2。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

from chimera_patrol.config import AppConfig
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.detect.worker import DetectWorker
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.ingest.camera import Camera, mask_source
from chimera_patrol.ingest.capture import CaptureThread, DetectQueue, FrameSlot
from chimera_patrol.ingest.discover import discover_cameras
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.rules.types import CandidateEvent
from chimera_patrol.store.repository import EventRepository

log = logging.getLogger("chimera_patrol.runtime")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _log_future(future: Future[Any]) -> None:
    try:
        future.result()
    except Exception:
        log.exception("候选事件落库失败")


class PatrolRuntime:
    """M1 的全部组件装配与生命周期管理。

    Args:
        config: 已校验的 AppConfig。
        repository: 事件存储。
        detector: 具备 detect_persons / detect_fire_smoke / detect_helmets 的检测器。
        loop: 运行 EventPipeline 的事件循环（需在另一线程中运行）。
        candidates: 可选的候选视频文件列表，缺省扫描 ``video.source_dir``。
        capture_factory: 可选的 VideoCapture 工厂，测试时注入假源。
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
    ) -> None:
        self._config = config
        self._repo = repository
        self._detector = detector
        self._loop = loop
        self._candidates = candidates
        self._capture_factory = capture_factory
        self._cameras: list[Camera] = []
        self._captures: list[CaptureThread] = []
        self._workers: list[DetectWorker] = []
        self._pipeline: EventPipeline | None = None

    @property
    def cameras(self) -> Sequence[Camera]:
        return tuple(self._cameras)

    def start(self) -> None:
        """发现 Camera，登记数据库，启动所有线程。"""
        cfg = self._config
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

        self._pipeline = EventPipeline(
            repository=self._repo,
            ring_buffers=ring_buffers,
            evidence=cfg.evidence,
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
        log.info("runtime started cameras=%d", len(self._cameras))

    def stop(self, timeout_s: float = 10.0) -> None:
        """停止所有采集与检测线程，并关闭流水线（取消后台片段任务）。"""
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
        self._captures.clear()
        self._workers.clear()

    # 内部回调（运行在采集线程或检测线程中）

    def _submit(self, candidate: CandidateEvent) -> None:
        if self._pipeline is None:
            raise RuntimeError("runtime is not started")
        future = asyncio.run_coroutine_threadsafe(self._pipeline.submit(candidate), self._loop)
        future.add_done_callback(_log_future)

    def _on_status_change(self, camera_id: str, status: str, ts_ms: int) -> None:
        try:
            self._repo.set_camera_status(camera_id, status, ts_ms)
        except Exception:
            log.exception("更新 Camera 状态失败 camera=%s", camera_id)