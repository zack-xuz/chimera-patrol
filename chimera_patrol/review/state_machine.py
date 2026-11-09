"""事件状态机：迁移合法性判断与状态迁移执行。

迁移规则与 design.md「事件状态机」及 event-contract.md 第 4.2 节一致。

- ``can_transition`` 为纯函数，只查表。
- ``EventStateMachine`` 读取当前状态，调用 ``can_transition`` 校验，然后通过
  ``EventRepository.transition_status`` 在同一事务内更新 ``events.status`` 并追加
  ``event_status_log``。成功后记录一条状态变化日志。

异常：

- ``InvalidTransition``：迁移不合法，或事件在读取后被其他请求改变。
- ``EventNotFound``（``LookupError`` 子类）：事件不存在。
- ``ValueError``：Inspector 的 ``operator`` 长度不在 1 至 32 字符之间。
"""

from __future__ import annotations

import time
from typing import Final

from chimera_patrol.logging_setup import log_status_change
from chimera_patrol.store.repository import (
    EVENT_STATUSES,
    EventRepository,
    StatusConflictError,
)

ACTOR_SYSTEM: Final = "system"
ACTOR_INSPECTOR: Final = "inspector"

OPERATOR_MIN_LENGTH: Final = 1
OPERATOR_MAX_LENGTH: Final = 32

PENDING_REVIEW: Final = "pending_review"
CONFIRMED: Final = "confirmed"
DISMISSED: Final = "dismissed"
NEEDS_HUMAN: Final = "needs_human"
REVIEW_FAILED: Final = "review_failed"

# 系统迁移：只能从 pending_review 发起。
_SYSTEM_TRANSITIONS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        (PENDING_REVIEW, CONFIRMED),
        (PENDING_REVIEW, DISMISSED),
        (PENDING_REVIEW, NEEDS_HUMAN),
        (PENDING_REVIEW, REVIEW_FAILED),
    }
)

# Inspector 改判：目标只能是 confirmed、dismissed、needs_human。
_INSPECTOR_TRANSITIONS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        (NEEDS_HUMAN, CONFIRMED),
        (NEEDS_HUMAN, DISMISSED),
        (REVIEW_FAILED, CONFIRMED),
        (REVIEW_FAILED, DISMISSED),
        (CONFIRMED, DISMISSED),
        (DISMISSED, CONFIRMED),
        (CONFIRMED, NEEDS_HUMAN),
        (DISMISSED, NEEDS_HUMAN),
    }
)


class InvalidTransition(Exception):
    """迁移不合法。API 层应映射为 409 ``invalid_transition``。"""

    def __init__(
        self,
        event_id: str,
        from_status: str | None,
        to_status: str,
        actor_type: str,
        detail: str | None = None,
    ) -> None:
        self.event_id = event_id
        self.from_status = from_status
        self.to_status = to_status
        self.actor_type = actor_type
        message = (
            f"invalid transition for event {event_id}: "
            f"{from_status!r} -> {to_status!r} by {actor_type}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class EventNotFound(LookupError):
    """指定 ID 的事件不存在。"""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        super().__init__(f"event not found: {event_id}")


def can_transition(from_status: str, to_status: str, actor_type: str) -> bool:
    """判断迁移是否合法。纯函数，不访问数据库。

    - ``system``：仅允许 ``pending_review`` 到 confirmed、dismissed、needs_human、review_failed。
    - ``inspector``：仅允许表中列出的改判对，目标限于 confirmed、dismissed、needs_human。
    - 同状态迁移、未知状态、未知 actor_type 均返回 False。
    """
    if from_status not in EVENT_STATUSES or to_status not in EVENT_STATUSES:
        return False
    if from_status == to_status:
        return False
    if actor_type == ACTOR_SYSTEM:
        return (from_status, to_status) in _SYSTEM_TRANSITIONS
    if actor_type == ACTOR_INSPECTOR:
        return (from_status, to_status) in _INSPECTOR_TRANSITIONS
    return False


def _validate_operator(operator: str) -> None:
    if not isinstance(operator, str):
        raise ValueError("operator must be a string")
    if not OPERATOR_MIN_LENGTH <= len(operator) <= OPERATOR_MAX_LENGTH:
        raise ValueError(
            f"operator must be {OPERATOR_MIN_LENGTH} to {OPERATOR_MAX_LENGTH} characters, "
            f"got {len(operator)}"
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


class EventStateMachine:
    """对事件执行状态迁移，并写入状态日志。

    ``repository`` 必须已经打开，并且数据库中已存在对应事件。
    """

    def __init__(self, repository: EventRepository) -> None:
        self._repo = repository

    def system_transition(
        self,
        event_id: str,
        to_status: str,
        actor: str,
        reason: str | None = None,
        *,
        now_ms: int | None = None,
    ) -> None:
        """以 ``system`` 身份迁移事件（例如 ``vlm_reviewer``、``sop_checker``）。

        ``actor`` 为非空的系统来源名称。``now_ms`` 为 UTC 毫秒，默认为当前时间。
        异常见模块说明。
        """
        self._transition(event_id, to_status, ACTOR_SYSTEM, actor, reason, now_ms)

    def override(
        self,
        event_id: str,
        to_status: str,
        operator: str,
        reason: str | None = None,
        *,
        now_ms: int | None = None,
    ) -> None:
        """以 ``inspector`` 身份改判。``operator`` 长度必须为 1 至 32 字符（假设 A9）。

        该值仅用于留痕，系统不做认证。``now_ms`` 为 UTC 毫秒，默认为当前时间。
        """
        _validate_operator(operator)
        self._transition(event_id, to_status, ACTOR_INSPECTOR, operator, reason, now_ms)

    def _transition(
        self,
        event_id: str,
        to_status: str,
        actor_type: str,
        actor: str,
        reason: str | None,
        now_ms: int | None,
    ) -> None:
        event = self._repo.get_event(event_id)
        if event is None:
            raise EventNotFound(event_id)
        current = event["status"]
        if not can_transition(current, to_status, actor_type):
            raise InvalidTransition(event_id, current, to_status, actor_type)

        timestamp = now_ms if now_ms is not None else _now_ms()
        try:
            self._repo.transition_status(
                event_id,
                from_status=current,
                to_status=to_status,
                actor_type=actor_type,
                actor=actor,
                reason=reason,
                now_ms=timestamp,
            )
        except StatusConflictError as exc:
            # 读取与写入之间状态被其他请求改变，本次迁移基于过期状态，不再合法。
            raise InvalidTransition(
                event_id, current, to_status, actor_type, detail=str(exc)
            ) from exc

        log_status_change(
            event_id=event_id,
            from_status=current,
            to_status=to_status,
            actor=actor,
            reason=reason,
        )
