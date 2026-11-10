"""SopChecker：SOP 合规检查的触发、采样、判定与结局（任务 15.8；需求 11.3、11.6、11.9、11.10）。

流程：

1. ``trigger`` 校验 SOP 存在、Camera 存在且适用、Camera 在线、同一 Camera 无进行中检查，
   计算采样时刻并写入 ``running`` 运行记录，立即返回。
2. 后台任务等待到窗口结束 ``T + window_after_s``，从 RingBuffer 取每个采样时刻的最近邻帧，
   保存采样帧，然后经 VLM_Adapter 的 ``check_sop`` 判定，结果经 ``normalize_sop_result`` 归一化。
3. 结局：
   - 成功且存在 fail：运行置为 completed，写入步骤结果，创建一条 ``sop_violation`` 事件
     （detection_source ``sop``，状态 pending_review，随后由系统迁移到 needs_human，actor ``sop_checker``）。
     关键帧取首个失败步骤对应的采样帧，缺省取中间采样帧。
   - 成功且无 fail：运行置为 completed，不创建事件。
   - 超时、适配器异常、响应非法、无帧：运行置为 check_failed_retry 并写入失败原因，
     不创建事件，不写入任何步骤结果。
4. 每次状态变化都经 notifier 推送 ``sop.run.updated``。

异常：``trigger`` 失败时抛出 ``SopTriggerError``，``code`` 与 design 错误码一致，
由 API 层映射为 HTTP 状态。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from chimera_patrol.config import EvidenceSection, SopSection
from chimera_patrol.evidence.ring_buffer import RingBuffer, select_frame
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.review.state_machine import NEEDS_HUMAN, EventStateMachine
from chimera_patrol.rules.types import CandidateEvent
from chimera_patrol.sop.loader import SopDefinition
from chimera_patrol.sop.normalize import NormalizedStep, SopResultError, normalize_sop_result
from chimera_patrol.sop.sampling import sample_times
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.sop_repo import NewSopRun, RunInProgressError, SopRepository, StepResult
from chimera_patrol.vlm.normalize import VLMResponseError
from chimera_patrol.vlm.protocol import SopCheckRequest, VLMAdapter

__all__ = ["SopChecker", "SopTriggerError", "SopNotifier", "ACTOR_SOP_CHECKER"]

ACTOR_SOP_CHECKER = "sop_checker"
SOP_EVENT_TYPE = "sop_violation"

_log = logging.getLogger("chimera_patrol.sop")

Frame = tuple[int, int, bytes]  # (index, ts_ms, jpeg)


class SopNotifier(Protocol):
    async def on_sop_run(self, run_id: str) -> None: ...

    async def on_event_updated(self, event_id: str) -> None: ...


class SopTriggerError(Exception):
    """SOP 触发被拒绝。``code`` 为 not_found、sop_not_applicable、camera_offline、run_in_progress 或 validation_error。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class _CheckFailed(Exception):
    """判定过程中的可预期失败，携带 failure_reason。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _new_run_id() -> str:
    return f"sop_{uuid.uuid4().hex[:12]}"


class SopChecker:
    """SOP 检查的运行时对象。方法都在事件循环中调用。"""

    def __init__(
        self,
        *,
        definitions: Sequence[SopDefinition],
        repository: EventRepository,
        sop_repo: SopRepository,
        pipeline: EventPipeline,
        state: EventStateMachine,
        adapter: VLMAdapter | None,
        ring_buffers: Mapping[str, RingBuffer],
        sop_cfg: SopSection,
        evidence_cfg: EvidenceSection,
        timeout_s: float,
        semaphore: asyncio.Semaphore,
        notifier: SopNotifier | None = None,
    ) -> None:
        self._definitions: dict[str, SopDefinition] = {}
        for definition in definitions:
            self._definitions[definition.id] = definition
        self._repo = repository
        self._sop_repo = sop_repo
        self._pipeline = pipeline
        self._state = state
        self._adapter = adapter
        self._ring_buffers = dict(ring_buffers)
        self._cfg = sop_cfg
        self._base_dir = Path(evidence_cfg.base_dir)
        self._timeout_s = float(timeout_s)
        self._semaphore = semaphore
        self._notifier = notifier
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def definitions(self) -> list[SopDefinition]:
        return list(self._definitions.values())

    def get_definition(self, sop_id: str) -> SopDefinition | None:
        return self._definitions.get(sop_id)

    # ------------------------------------------------------------------ 触发

    async def trigger(
        self,
        *,
        sop_id: str,
        camera_id: str,
        operator: str,
        retry_of: str | None = None,
    ) -> str:
        """校验并创建一次 running 检查，返回运行 ID。后台任务随后执行判定。

        Raises:
            SopTriggerError: 校验失败，``code`` 见类说明。
        """
        sop = self._definitions.get(sop_id)
        if sop is None:
            raise SopTriggerError("not_found", f"sop {sop_id} not found")

        camera = await asyncio.to_thread(self._repo.get_camera, camera_id)
        if camera is None:
            raise SopTriggerError("not_found", f"camera {camera_id} not found")
        if "*" not in sop.camera_ids and camera_id not in sop.camera_ids:
            raise SopTriggerError(
                "sop_not_applicable", f"sop {sop_id} does not apply to camera {camera_id}"
            )

        if retry_of is not None:
            previous = await asyncio.to_thread(self._sop_repo.get_run, retry_of)
            if previous is None:
                raise SopTriggerError("not_found", f"sop run {retry_of} not found")
            prev = previous.row
            if prev["status"] != "check_failed_retry":
                raise SopTriggerError(
                    "validation_error", f"retry_of {retry_of} is not check_failed_retry"
                )
            if prev["sop_id"] != sop_id or prev["camera_id"] != camera_id:
                raise SopTriggerError(
                    "validation_error", "retry_of must refer to the same sop and camera"
                )

        if camera["status"] != "online":
            raise SopTriggerError("camera_offline", f"camera {camera_id} is offline")

        running = await asyncio.to_thread(self._sop_repo.find_running, camera_id)
        if running is not None:
            raise SopTriggerError(
                "run_in_progress", f"camera {camera_id} already has a running SOP check"
            )

        triggered = _now_ms()
        before_ms = round(self._cfg.window_before_s * 1000)
        after_ms = round(self._cfg.window_after_s * 1000)
        times = sample_times(
            triggered,
            self._cfg.window_before_s,
            self._cfg.window_after_s,
            self._cfg.sample_count,
            self._cfg.endpoints_inclusive,
        )
        run_id = _new_run_id()
        new_run = NewSopRun(
            id=run_id,
            sop_id=sop.id,
            sop_name=sop.name,
            camera_id=camera_id,
            operator=operator,
            triggered_ms=triggered,
            window_start_ms=triggered - before_ms,
            window_end_ms=triggered + after_ms,
            sample_count=self._cfg.sample_count,
            sample_times_ms=times,
            created_ms=triggered,
            retry_of=retry_of,
        )
        try:
            await asyncio.to_thread(self._sop_repo.insert_run, new_run)
        except RunInProgressError as exc:
            raise SopTriggerError("run_in_progress", str(exc)) from exc

        await self._notify_run(run_id)
        task: asyncio.Task[None] = asyncio.ensure_future(
            self._execute(run_id, sop, camera_id, new_run.window_start_ms, new_run.window_end_ms, times)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run_id

    async def close(self) -> None:
        """取消未完成的检查任务并等待结束。"""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------ 执行

    async def _execute(
        self,
        run_id: str,
        sop: SopDefinition,
        camera_id: str,
        window_start: int,
        window_end: int,
        times: list[int],
    ) -> None:
        try:
            await asyncio.sleep(max(0.0, (window_end - _now_ms()) / 1000.0))
            frames = await asyncio.to_thread(self._collect_frames, camera_id, window_start, window_end, times)
            if not frames:
                raise _CheckFailed("no_frames")
            await asyncio.to_thread(self._save_sample_frames, run_id, frames)

            steps = await self._judge(run_id, sop, frames)
            await self._complete(run_id, sop, camera_id, window_start, window_end, steps, frames)
        except asyncio.CancelledError:
            raise
        except _CheckFailed as exc:
            await self._fail(run_id, exc.reason)
        except Exception:
            _log.exception("SOP 检查异常 run_id=%s", run_id)
            await self._fail(run_id, "vlm_error")

    def _collect_frames(
        self, camera_id: str, window_start: int, window_end: int, times: list[int]
    ) -> list[Frame]:
        """在工作线程中为每个采样时刻选取最近邻帧。窗口内无帧时返回空列表。"""
        ring = self._ring_buffers.get(camera_id)
        if ring is None:
            return []
        snapshot = ring.snapshot(window_start, window_end)
        if not snapshot:
            return []
        frames: list[Frame] = []
        for index, when in enumerate(times):
            ts, data = select_frame(snapshot, when)
            frames.append((index, ts, data))
        return frames

    def _save_sample_frames(self, run_id: str, frames: Sequence[Frame]) -> None:
        """保存采样帧到 {base_dir}/sop/{run_id}/frame_{i}.jpg，失败只记日志。"""
        out_dir = self._base_dir / "sop" / run_id
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, _ts, data in frames:
                (out_dir / f"frame_{index}.jpg").write_bytes(data)
        except OSError:
            _log.exception("SOP 采样帧保存失败 run_id=%s", run_id)

    async def _judge(self, run_id: str, sop: SopDefinition, frames: Sequence[Frame]) -> list[NormalizedStep]:
        """调用 VLM 判定并归一化。失败时抛出 _CheckFailed。"""
        if self._adapter is None:
            raise _CheckFailed("vlm_error")
        request = SopCheckRequest(
            run_id=run_id,
            sop_name=sop.name,
            steps=[(step.id, step.description, step.criteria) for step in sop.steps],
            frames=list(frames),
        )
        try:
            async with self._semaphore:
                raw = await asyncio.wait_for(self._adapter.check_sop(request), timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise _CheckFailed("vlm_timeout") from exc
        except VLMResponseError as exc:
            raise _CheckFailed("response_invalid") from exc
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("SOP 判定调用失败 run_id=%s", run_id)
            raise _CheckFailed("vlm_error") from None

        try:
            return normalize_sop_result(sop, raw)
        except SopResultError as exc:
            _log.warning("SOP 结果结构非法 run_id=%s error=%s", run_id, exc)
            raise _CheckFailed("response_invalid") from exc

    async def _complete(
        self,
        run_id: str,
        sop: SopDefinition,
        camera_id: str,
        window_start: int,
        window_end: int,
        steps: list[NormalizedStep],
        frames: Sequence[Frame],
    ) -> None:
        results = [
            StepResult(
                index=step.index,
                step_id=step.step_id,
                description=step.description,
                status=step.status,
                evidence=step.evidence,
                approximate=step.approximate,
            )
            for step in steps
        ]
        await asyncio.to_thread(self._sop_repo.finish_completed, run_id, results, _now_ms())
        await self._notify_run(run_id)

        failed = [step for step in steps if step.status == "fail"]
        if not failed:
            return

        key_index = failed[0].evidence_frame
        if key_index is None or not 0 <= key_index < len(frames):
            key_index = len(frames) // 2
        candidate = CandidateEvent(
            camera_id=camera_id,
            event_type=SOP_EVENT_TYPE,
            detection_source="sop",
            start_ms=window_start,
            end_ms=window_end,
            cv_confidence=None,
            track_id=None,
            roi_id=None,
            roi_type=None,
            detail={
                "sop_id": sop.id,
                "sop_name": sop.name,
                "failed_steps": [step.step_id for step in failed],
            },
            trajectory=[],
            overlay=[],
            sop_run_id=run_id,
        )
        result = await self._pipeline.submit(candidate, keyframe_jpeg=frames[key_index][2], review=False)
        if result.status != "pending_review":
            _log.error("SOP 不合规事件未生成 run_id=%s status=%s", run_id, result.status)
            return
        await asyncio.to_thread(
            self._state.system_transition,
            result.event_id,
            NEEDS_HUMAN,
            ACTOR_SOP_CHECKER,
            "SOP 不合规",
        )
        if self._notifier is not None:
            await self._notifier.on_event_updated(result.event_id)

    async def _fail(self, run_id: str, reason: str) -> None:
        """写入 check_failed_retry。仅影响仍为 running 的记录，已完成的记录不受影响。"""
        try:
            changed = await asyncio.to_thread(self._sop_repo.finish_check_failed, run_id, reason, _now_ms())
        except Exception:
            _log.exception("写入 SOP 检查失败状态出错 run_id=%s", run_id)
            return
        if changed:
            _log.warning("SOP 检查失败待重试 run_id=%s reason=%s", run_id, reason)
        await self._notify_run(run_id)

    async def _notify_run(self, run_id: str) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.on_sop_run(run_id)
        except Exception:
            _log.exception("sop.run.updated 推送失败 run_id=%s", run_id)

