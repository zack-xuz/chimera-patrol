"""VLM 复核任务 VlmReviewer（任务 10.5）。

对单个 pending_review 事件执行异步复核（design: VLM 复核层、线程与异步模型）：

- ``asyncio.Semaphore(max_concurrency)`` 限制同时在途的适配器调用数。
- ``asyncio.wait_for(..., timeout_s)`` 只包裹适配器调用本身。
- 成功时写入 vlm_* 字段，按 ``decide_status`` 迁移状态。
- 超时、适配器异常、响应非法、关键帧缺失时迁移为 ``review_failed``。
- 复核关闭时不调用适配器，直接迁移为 ``needs_human``（actor ``review_switch``）。

``review`` 不会把普通异常抛给调用方；``asyncio.CancelledError`` 原样重新抛出且不改变状态。
数据库与状态机调用都通过 ``asyncio.to_thread`` 执行，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from chimera_patrol.logging_setup import log_vlm_request, log_vlm_response, redact, truncate
from chimera_patrol.review.decision import decide_status
from chimera_patrol.review.state_machine import NEEDS_HUMAN, REVIEW_FAILED, EventStateMachine
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.vlm.normalize import VLMResponseError
from chimera_patrol.vlm.protocol import ReviewRequest, VLMAdapter

ACTOR_REVIEWER = "vlm_reviewer"
ACTOR_REVIEW_SWITCH = "review_switch"

ERROR_TIMEOUT = "timeout"
ERROR_RESPONSE_INVALID = "response_invalid"
ERROR_KEYFRAME_MISSING = "keyframe_missing"
VLM_ERROR_PREFIX = "vlm_error: "
VLM_ERROR_MAX_CHARS = 200

_logger = logging.getLogger("chimera_patrol.review")


@dataclass(frozen=True)
class ReviewOutcome:
    """复核结束后的结果。

    ``status`` 为写入事件的最终状态；若状态无法写入则为 None。
    ``error`` 为失败原因（与 vlm_error 相同），成功或复核关闭时为 None。
    """

    event_id: str
    status: str | None
    error: str | None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _vlm_error_text(exc: BaseException) -> str:
    """把任意异常转为 ``vlm_error: <类型>: <信息>``，单行且不超过 200 字符。"""
    detail = " ".join(redact(str(exc)).split())
    text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return truncate(f"{VLM_ERROR_PREFIX}{text}", VLM_ERROR_MAX_CHARS)


class VlmReviewer:
    """异步 VLM 复核。每个实例共享一个并发信号量。

    Args:
        adapter: VLM_Adapter 实现，只使用 ``review_event``。
        state_machine: 用于系统迁移的 EventStateMachine。
        repository: 用于写入 vlm_* 字段的 EventRepository。
        timeout_s: 单次适配器调用的超时秒数（vlm.timeout_s）。
        max_concurrency: 同时在途的适配器调用数上限（vlm.max_concurrency）。
        description_max_chars: 发送给 VLM 的描述长度上限（vlm.description_max_chars）。
        auto_decision_threshold: 自动判定置信度阈值（vlm.auto_decision_threshold）。
        review_enabled: 为 False 时不调用适配器（vlm.review_enabled）。
    """

    def __init__(
        self,
        adapter: VLMAdapter,
        state_machine: EventStateMachine,
        repository: EventRepository,
        *,
        timeout_s: float,
        max_concurrency: int,
        description_max_chars: int,
        auto_decision_threshold: float,
        review_enabled: bool,
    ) -> None:
        if isinstance(timeout_s, bool) or not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError(f"max_concurrency must be an integer >= 1, got {max_concurrency!r}")
        if (
            isinstance(description_max_chars, bool)
            or not isinstance(description_max_chars, int)
            or description_max_chars < 1
        ):
            raise ValueError(
                f"description_max_chars must be an integer >= 1, got {description_max_chars!r}"
            )
        if isinstance(auto_decision_threshold, bool) or not 0.0 <= float(auto_decision_threshold) <= 1.0:
            raise ValueError(
                f"auto_decision_threshold must be within [0, 1], got {auto_decision_threshold!r}"
            )

        self._adapter = adapter
        self._state = state_machine
        self._repo = repository
        self._timeout_s = float(timeout_s)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._description_max_chars = description_max_chars
        self._threshold = float(auto_decision_threshold)
        self._review_enabled = bool(review_enabled)

    async def review(
        self,
        event_id: str,
        *,
        event_type: str,
        camera_name: str,
        roi_name: str | None,
        cv_confidence: float,
        image_jpeg: bytes | None,
    ) -> ReviewOutcome:
        """复核一个 pending_review 事件并写回结果。

        普通异常在内部被捕获并记录，返回 ``status=None`` 的结果。
        ``asyncio.CancelledError`` 会原样抛出，且不修改事件状态。
        """
        try:
            return await self._review(
                event_id,
                event_type=event_type,
                camera_name=camera_name,
                roi_name=roi_name,
                cv_confidence=cv_confidence,
                image_jpeg=image_jpeg,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - review() must never raise to the pipeline
            detail = _vlm_error_text(exc)
            _logger.error("review aborted id=%s error=%s", event_id, detail)
            return ReviewOutcome(event_id=event_id, status=None, error=detail)

    async def _review(
        self,
        event_id: str,
        *,
        event_type: str,
        camera_name: str,
        roi_name: str | None,
        cv_confidence: float,
        image_jpeg: bytes | None,
    ) -> ReviewOutcome:
        if not self._review_enabled:
            await asyncio.to_thread(self._repo.update_vlm_result, event_id, review_skipped=True)
            await asyncio.to_thread(
                self._state.system_transition,
                event_id,
                NEEDS_HUMAN,
                ACTOR_REVIEW_SWITCH,
                "review disabled",
            )
            return ReviewOutcome(event_id=event_id, status=NEEDS_HUMAN, error=None)

        if not image_jpeg:
            return await self._fail(event_id, ERROR_KEYFRAME_MISSING)

        request = ReviewRequest(
            event_id=event_id,
            event_type=event_type,
            camera_name=camera_name,
            roi_name=roi_name,
            cv_confidence=cv_confidence,
            image_jpeg=image_jpeg,
            description_max_chars=self._description_max_chars,
        )
        log_vlm_request(event_id=event_id, image_bytes=len(image_jpeg))
        started = time.monotonic()

        try:
            async with self._semaphore:
                verdict = await asyncio.wait_for(
                    self._adapter.review_event(request), timeout=self._timeout_s
                )
        except asyncio.TimeoutError:
            self._log_failure(event_id, started, ERROR_TIMEOUT)
            return await self._fail(event_id, ERROR_TIMEOUT)
        except VLMResponseError:
            self._log_failure(event_id, started, ERROR_RESPONSE_INVALID)
            return await self._fail(event_id, ERROR_RESPONSE_INVALID)
        except Exception as exc:  # noqa: BLE001 - adapter failures are recorded, not raised
            error = _vlm_error_text(exc)
            self._log_failure(event_id, started, error)
            return await self._fail(event_id, error)

        duration_ms = int((time.monotonic() - started) * 1000)
        log_vlm_response(
            event_id=event_id,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            duration_ms=duration_ms,
            summary=verdict.description,
        )

        status = decide_status(verdict.verdict, verdict.confidence, self._threshold)
        await asyncio.to_thread(
            self._repo.update_vlm_result,
            event_id,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            description=verdict.description,
            suggestion=verdict.suggestion,
            error=None,
            reviewed_ms=_now_ms(),
        )
        await asyncio.to_thread(self._state.system_transition, event_id, status, ACTOR_REVIEWER, None)
        return ReviewOutcome(event_id=event_id, status=status, error=None)

    def _log_failure(self, event_id: str, started: float, error: str) -> None:
        duration_ms = int((time.monotonic() - started) * 1000)
        log_vlm_response(
            event_id=event_id,
            verdict=None,
            confidence=None,
            duration_ms=duration_ms,
            summary=error,
        )

    async def _fail(self, event_id: str, error: str) -> ReviewOutcome:
        """写入 vlm_error 并迁移到 review_failed，reason 与 vlm_error 相同。"""
        await asyncio.to_thread(self._repo.update_vlm_result, event_id, error=error)
        await asyncio.to_thread(
            self._state.system_transition, event_id, REVIEW_FAILED, ACTOR_REVIEWER, error
        )
        return ReviewOutcome(event_id=event_id, status=REVIEW_FAILED, error=error)
