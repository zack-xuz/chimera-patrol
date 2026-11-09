"""EventPipeline：把 Rule_Engine 产出的 CandidateEvent 落地为事件（任务 6.1）。

这里是 asyncio 事件循环一侧的编排层。DetectWorker 线程通过
``run_coroutine_threadsafe`` 调用 ``submit``，线程切换的接线属于任务 6.2，不在本模块。

``submit`` 的顺序：

1. 生成 ``evt_`` + 12 位十六进制的事件 ID（uuid4 的前 12 位）。
2. 在线程中保存关键帧（标注版与原图版）。
3. 在线程中落库，状态为 ``pending_review``，``clip_status`` 为 ``pending``。
4. 记录事件生成日志。
5. 调用 ``on_created(summary)``，``summary`` 含 id、camera_id、event_type、status。
6. 启动后台片段任务：等待到 ``end_ms + clip_post_s`` 的墙钟时刻后，在线程中截取并编码片段，
   再更新 ``clip_path`` 与 ``clip_status``，最后调用 ``on_clip_done(event_id, status)``。
7. 调用 ``on_review(event_id)``，由 M2 决定复核路径。本模块不调用任何 VLM。

关键帧缺失的处理：``events`` 表要求关键帧路径非空，因此无法保存关键帧时事件不能落库。
``save_keyframe`` 抛出 ``KeyframeUnavailable`` 时，本模块记录带事件上下文的 ERROR 日志，
不写入事件，并返回 ``status="dropped_no_keyframe"``。

错误边界：

- 关键帧的非 ``KeyframeUnavailable`` 异常与 ``insert_event`` 的异常：记录日志后原样抛给调用方。
- ``on_created``、``on_review``、``on_clip_done`` 钩子异常：记录日志后吞掉。事件已落库，
  钩子失败不应让已存在的事件消失或改变其状态。
- 片段任务中的任何失败：记录日志，``clip_status`` 置为 ``failed``，事件状态不变（需求 4.3）。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chimera_patrol.config import EvidenceSection
from chimera_patrol.evidence.clip import save_clip_safely
from chimera_patrol.evidence.keyframe import KeyframeUnavailable, save_keyframe
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.logging_setup import log_event_generated
from chimera_patrol.rules.types import CandidateEvent
from chimera_patrol.store.repository import EventRepository, NewEvent

__all__ = ["EventPipeline", "PipelineResult", "STATUS_PENDING_REVIEW", "STATUS_DROPPED_NO_KEYFRAME"]

STATUS_PENDING_REVIEW = "pending_review"
STATUS_DROPPED_NO_KEYFRAME = "dropped_no_keyframe"

_log = logging.getLogger("chimera_patrol.pipeline")

OnCreated = Callable[[dict[str, Any]], Awaitable[None]]
OnClipDone = Callable[[str, str], Awaitable[None]]
OnReview = Callable[[str], Awaitable[None]]
ClipSaver = Callable[..., tuple[str, str]]


@dataclass(frozen=True)
class PipelineResult:
    """``submit`` 的返回值。

    ``status`` 为 ``pending_review`` 或 ``dropped_no_keyframe``。
    ``detail`` 在事件被丢弃时给出原因，否则为 None。
    """

    event_id: str
    status: str
    detail: str | None = None


async def _no_op_created(_: dict[str, Any]) -> None:
    return None


async def _no_op_clip_done(_: str, __: str) -> None:
    return None


async def _no_op_review(_: str) -> None:
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


class EventPipeline:
    """事件落库编排器。一个进程内一个实例，所有方法都在事件循环中调用。"""

    def __init__(
        self,
        *,
        repository: EventRepository,
        ring_buffers: Mapping[str, RingBuffer],
        evidence: EvidenceSection,
        on_created: OnCreated | None = None,
        on_clip_done: OnClipDone | None = None,
        on_review: OnReview | None = None,
        clip_saver: ClipSaver = save_clip_safely,
    ) -> None:
        """
        Args:
            repository: 事件存储。阻塞调用经 ``asyncio.to_thread`` 执行。
            ring_buffers: 摄像头 ID 到该路 RingBuffer 的映射。
            evidence: ``evidence`` 配置段，提供 base_dir、片段前后时长与关键帧 JPEG 质量。
            on_created, on_clip_done, on_review: 可选异步钩子，默认为空操作。
            clip_saver: 片段保存函数，签名同 ``save_clip_safely``。默认值为真实实现，
                测试可替换它以注入失败。
        """
        self._repo = repository
        self._ring_buffers = ring_buffers
        self._evidence = evidence
        self._on_created: OnCreated = on_created or _no_op_created
        self._on_clip_done: OnClipDone = on_clip_done or _no_op_clip_done
        self._on_review: OnReview = on_review or _no_op_review
        self._clip_saver = clip_saver
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def submit(self, candidate: CandidateEvent) -> PipelineResult:
        """处理一个候选事件，返回 ``PipelineResult``。详见模块文档的顺序说明。"""
        if self._closed:
            raise RuntimeError("EventPipeline is closed")

        event_id = _new_event_id()
        ring = self._ring_buffers.get(candidate.camera_id)

        try:
            annotated_rel, raw_rel = await asyncio.to_thread(
                self._save_keyframes, event_id, candidate, ring
            )
        except KeyframeUnavailable as exc:
            _log.error(
                "event dropped, no keyframe event_id=%s camera_id=%s event_type=%s "
                "start_ms=%s end_ms=%s reason=%s",
                event_id,
                candidate.camera_id,
                candidate.event_type,
                candidate.start_ms,
                candidate.end_ms,
                exc,
            )
            return PipelineResult(event_id, STATUS_DROPPED_NO_KEYFRAME, str(exc))
        except Exception:
            _log.exception(
                "keyframe save failed event_id=%s camera_id=%s event_type=%s",
                event_id,
                candidate.camera_id,
                candidate.event_type,
            )
            raise

        new_event = NewEvent(
            id=event_id,
            camera_id=candidate.camera_id,
            event_type=candidate.event_type,
            detection_source=candidate.detection_source,
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
            keyframe_annotated_path=annotated_rel,
            keyframe_raw_path=raw_rel,
            created_ms=_now_ms(),
            cv_confidence=candidate.cv_confidence,
            track_id=candidate.track_id,
            roi_id=candidate.roi_id,
            roi_type=candidate.roi_type,
            cv_detail=candidate.detail,
            trajectory=candidate.trajectory,
            sop_run_id=candidate.sop_run_id,
        )
        try:
            await asyncio.to_thread(self._repo.insert_event, new_event)
        except Exception:
            _log.exception(
                "insert_event failed event_id=%s camera_id=%s event_type=%s",
                event_id,
                candidate.camera_id,
                candidate.event_type,
            )
            raise

        log_event_generated(
            event_id=event_id,
            camera_id=candidate.camera_id,
            event_type=candidate.event_type,
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
        )

        await self._run_hook(
            "on_created",
            event_id,
            self._on_created(
                {
                    "id": event_id,
                    "camera_id": candidate.camera_id,
                    "event_type": candidate.event_type,
                    "status": STATUS_PENDING_REVIEW,
                }
            ),
        )

        self._spawn(self._clip_job(event_id, candidate.camera_id, candidate.end_ms, ring))

        await self._run_hook("on_review", event_id, self._on_review(event_id))

        return PipelineResult(event_id, STATUS_PENDING_REVIEW)

    async def close(self) -> None:
        """取消所有未完成的后台片段任务并等待其结束。之后 ``submit`` 会抛出 RuntimeError。"""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # 内部实现

    def _save_keyframes(
        self, event_id: str, candidate: CandidateEvent, ring: RingBuffer | None
    ) -> tuple[str, str]:
        """在工作线程中运行。没有该路的 RingBuffer 时视为无关键帧。"""
        if ring is None:
            raise KeyframeUnavailable(f"no ring buffer registered for camera {candidate.camera_id}")
        return save_keyframe(
            ring,
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
            overlay=candidate.overlay,
            base_dir=Path(self._evidence.base_dir),
            jpeg_quality=self._evidence.ring_jpeg_quality,
            camera_id=candidate.camera_id,
            event_id=event_id,
        )

    def _spawn(self, coro: Awaitable[None]) -> None:
        task: asyncio.Task[None] = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _clip_job(
        self,
        event_id: str,
        camera_id: str,
        end_ms: int,
        ring: RingBuffer | None,
    ) -> None:
        """后台任务：等待片段窗口结束后保存片段并更新事件。失败只记日志，不影响事件状态。"""
        try:
            if ring is None:
                raise RuntimeError(f"no ring buffer registered for camera {camera_id}")

            ready_at_ms = end_ms + self._evidence.clip_post_s * 1000
            await asyncio.sleep(max(0.0, (ready_at_ms - _now_ms()) / 1000.0))

            try:
                status, detail = await asyncio.to_thread(
                    self._clip_saver,
                    ring,
                    evidence=self._evidence,
                    camera_id=camera_id,
                    event_id=event_id,
                    end_ms=end_ms,
                )
            except Exception as exc:
                _log.error(
                    "clip saver raised event_id=%s camera_id=%s error=%s: %s",
                    event_id,
                    camera_id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                status, detail = "failed", f"{type(exc).__name__}: {exc}"

            if status == "ready":
                await asyncio.to_thread(
                    self._repo.update_evidence,
                    event_id,
                    clip_path=detail,
                    clip_status="ready",
                )
            else:
                await asyncio.to_thread(
                    self._repo.update_evidence, event_id, clip_status="failed"
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception(
                "clip job failed event_id=%s camera_id=%s", event_id, camera_id
            )
            return

        await self._run_hook("on_clip_done", event_id, self._on_clip_done(event_id, status))

    async def _run_hook(self, name: str, event_id: str, awaitable: Awaitable[None]) -> None:
        """运行一个钩子。异常记录后吞掉，取消异常照常传播。"""
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("hook %s failed event_id=%s", name, event_id)
