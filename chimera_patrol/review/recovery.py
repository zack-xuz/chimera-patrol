"""服务重启恢复：把遗留的 pending_review 事件迁移为 review_failed（假设 A10）。

服务重启后，内存中的复核任务已经丢失，数据库中仍为 pending_review 的事件无人处理。
本模块在启动时把它们迁移为 review_failed，原因为 ``service_restart``，
状态日志的 actor 为 ``startup_recovery``，进入人工队列。

- 只处理 ``pending_review`` 事件，其他状态的事件不做任何改动。
- 只写入 ``vlm_error``，并通过状态机迁移 ``status``，其他字段保持不变。
- 幂等：恢复后的事件不再是 pending_review，再次运行返回 0。
- 每次迁移的日志由 ``EventStateMachine`` 写入，本模块不重复记录。

SOP 检查的恢复（running 置为 check_failed_retry）不在本模块内，属于任务 15.7。
"""

from __future__ import annotations

import logging
import time
from typing import Final

from chimera_patrol.review.state_machine import (
    PENDING_REVIEW,
    REVIEW_FAILED,
    EventStateMachine,
    InvalidTransition,
)
from chimera_patrol.store.repository import LIST_LIMIT_MAX, EventRepository
from chimera_patrol.store.sop_repo import SopRepository

RECOVERY_ACTOR: Final = "startup_recovery"
RECOVERY_REASON: Final = "service_restart"

_logger = logging.getLogger(__name__)


def _collect_pending_ids(repository: EventRepository) -> list[str]:
    """分页读取全部 pending_review 事件 ID。

    先完整收集再迁移，避免迁移过程中结果集缩小导致分页跳过记录。
    """
    ids: list[str] = []
    offset = 0
    while True:
        page, total = repository.list_events(
            status=PENDING_REVIEW, limit=LIST_LIMIT_MAX, offset=offset
        )
        ids.extend(event["id"] for event in page)
        offset += len(page)
        if not page or offset >= total:
            return ids


def recover_pending_reviews(
    repository: EventRepository,
    state_machine: EventStateMachine,
    now_ms: int | None = None,
) -> int:
    """恢复遗留的 pending_review 事件，返回本次实际恢复的事件数。

    对每个 pending_review 事件：

    1. 将 ``vlm_error`` 设为 ``service_restart``（仅当状态仍为 pending_review）。
    2. 通过 ``system_transition`` 迁移为 ``review_failed``，actor 为 ``startup_recovery``。

    若某事件在收集之后、迁移之前已被其他流程改变状态，跳过该事件并记录警告，
    不计入返回值。``now_ms`` 为 UTC 毫秒，默认为当前时间。
    """
    recovered = 0
    for event_id in _collect_pending_ids(repository):
        # 先写原因、后迁移状态：若中途崩溃，事件仍为 pending_review，下次启动会再次恢复。
        repository.update_vlm_error(
            event_id, RECOVERY_REASON, from_status=PENDING_REVIEW
        )
        try:
            state_machine.system_transition(
                event_id,
                REVIEW_FAILED,
                actor=RECOVERY_ACTOR,
                reason=RECOVERY_REASON,
                now_ms=now_ms,
            )
        except InvalidTransition:
            _logger.warning(
                "skip recovery for event %s: status changed during recovery", event_id
            )
            continue
        recovered += 1
    return recovered


def recover_running_sop_runs(sop_repo: SopRepository, now_ms: int | None = None) -> int:
    """启动恢复（15.7）：把遗留的 running SOP 检查置为 check_failed_retry，原因 service_restart。

    内存中的检查任务已随进程结束，数据库中的 running 记录无人处理。
    置为 check_failed_retry 后，Inspector 可以通过 retry_of 重新触发。返回恢复的数量。
    """
    timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
    count = sop_repo.recover_running(timestamp)
    if count:
        _logger.info("recovered %d running SOP checks as check_failed_retry", count)
    return count
