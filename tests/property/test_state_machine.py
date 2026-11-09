"""Property 15: 状态迁移合法性与审计（任务 10.4）。

**Validates: Requirements 6.1, 6.6**

对任意事件及任意由系统迁移与 Inspector 改判混合组成的操作序列：

- 非法迁移被拒绝，且不改变事件状态、不追加日志；
- 每次操作后 ``events.status`` 与模型一致，结束时等于最后一条日志的 ``to_status``；
- 首条日志为 ``NULL -> pending_review``，相邻日志满足 ``log[i].to_status == log[i+1].from_status``；
- 人工改判日志包含 operator、时间戳、原状态与新状态。

合法性判定使用本文件内的契约表（字面量），不导入 ``state_machine.can_transition``，
以免测试与被测代码共享同一份实现。
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Final

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chimera_patrol.review.state_machine import EventStateMachine, InvalidTransition
from chimera_patrol.store.repository import EVENT_STATUSES, EventRepository, NewEvent

# 契约表：与 design.md「事件状态机」一致，独立于被测代码书写。
CONTRACT_SYSTEM: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("pending_review", "confirmed"),
        ("pending_review", "dismissed"),
        ("pending_review", "needs_human"),
        ("pending_review", "review_failed"),
    }
)
CONTRACT_INSPECTOR: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("needs_human", "confirmed"),
        ("needs_human", "dismissed"),
        ("review_failed", "confirmed"),
        ("review_failed", "dismissed"),
        ("confirmed", "dismissed"),
        ("dismissed", "confirmed"),
        ("confirmed", "needs_human"),
        ("dismissed", "needs_human"),
    }
)

CAMERA_ID: Final = "cam1"
EVENT_ID: Final = "evt_000000000001"
CREATED_MS: Final = 1_000
BASE_TRANSITION_MS: Final = 2_000
SYSTEM_KIND: Final = "system"
INSPECTOR_KIND: Final = "inspector"

operations = st.lists(
    st.tuples(
        st.sampled_from([SYSTEM_KIND, INSPECTOR_KIND]),
        st.sampled_from(EVENT_STATUSES),
        st.text(min_size=1, max_size=32),
    ),
    min_size=1,
    max_size=25,
)


def _expected_legal(current: str, kind: str, to_status: str) -> bool:
    table = CONTRACT_SYSTEM if kind == SYSTEM_KIND else CONTRACT_INSPECTOR
    return (current, to_status) in table


def _new_event() -> NewEvent:
    return NewEvent(
        id=EVENT_ID,
        camera_id=CAMERA_ID,
        event_type="intrusion",
        detection_source="yolo",
        start_ms=1_000,
        end_ms=1_000,
        keyframe_annotated_path="annotated.jpg",
        keyframe_raw_path="raw.jpg",
        created_ms=CREATED_MS,
    )


def _read_status_log(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT from_status, to_status, actor_type, actor, created_ms "
            "FROM event_status_log WHERE event_id = ? ORDER BY id",
            (EVENT_ID,),
        ).fetchall()
    finally:
        conn.close()


def _read_event_status(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT status FROM events WHERE id = ?", (EVENT_ID,)).fetchone()[0]
    finally:
        conn.close()


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(ops=operations)
def test_state_machine_legality_and_audit(ops: list[tuple[str, str, str]]) -> None:
    """Property 15: 状态迁移合法性与审计。

    **Validates: Requirements 6.1, 6.6**
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "state.db"
        repo = EventRepository(db_path)
        try:
            # 外键要求的摄像头行，写法与 tests/unit/test_state_machine.py 相同。
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
                "VALUES (?, 'Camera 1', 'file', 'clip.mp4', 0)",
                (CAMERA_ID,),
            )
            conn.commit()
            conn.close()

            repo.insert_event(_new_event())
            machine = EventStateMachine(repo)

            model_status = "pending_review"
            # 模型记录的已接受迁移：(from, to, actor_type, actor, now_ms)
            accepted: list[tuple[str | None, str, str, str, int]] = [
                (None, "pending_review", "system", "cv_pipeline", CREATED_MS)
            ]

            for index, (kind, to_status, operator) in enumerate(ops):
                now_ms = BASE_TRANSITION_MS + index
                expected_ok = _expected_legal(model_status, kind, to_status)

                try:
                    if kind == SYSTEM_KIND:
                        machine.system_transition(
                            EVENT_ID, to_status, operator, now_ms=now_ms
                        )
                    else:
                        machine.override(EVENT_ID, to_status, operator, now_ms=now_ms)
                    actually_ok = True
                except InvalidTransition:
                    actually_ok = False

                assert actually_ok == expected_ok, (
                    f"op {index} ({kind}, {model_status!r} -> {to_status!r}, "
                    f"operator={operator!r}): expected legal={expected_ok}, "
                    f"got legal={actually_ok}"
                )

                if expected_ok:
                    accepted.append(
                        (model_status, to_status, kind if kind == INSPECTOR_KIND else "system",
                         operator, now_ms)
                    )
                    model_status = to_status

                # 每次操作后，events.status 必须与模型一致（拒绝时保持不变）。
                assert _read_event_status(db_path) == model_status, (
                    f"after op {index}: stored status {_read_event_status(db_path)!r} "
                    f"!= model {model_status!r}"
                )

            log = _read_status_log(db_path)

            # 日志与已接受迁移逐行一致：拒绝的操作不写日志。
            assert len(log) == len(accepted), f"log rows {len(log)} != accepted {len(accepted)}"
            for row, expected in zip(log, accepted, strict=True):
                from_status, to_status, actor_type, actor, created_ms = row
                exp_from, exp_to, exp_actor_type, exp_actor, exp_ms = expected
                assert (from_status, to_status, actor_type, actor, created_ms) == (
                    exp_from,
                    exp_to,
                    exp_actor_type,
                    exp_actor,
                    exp_ms,
                )

            # 首条日志为 NULL -> pending_review。
            assert log[0][0] is None and log[0][1] == "pending_review"

            # 相邻日志首尾相接。
            for prev, nxt in zip(log, log[1:], strict=False):
                assert prev[1] == nxt[0], f"adjacent mismatch: {prev} then {nxt}"

            # 事件当前状态等于最后一条日志的 to_status。
            assert _read_event_status(db_path) == log[-1][1]

            # 人工改判日志包含 operator、时间戳、原状态与新状态。
            for from_status, to_status, actor_type, actor, created_ms in log[1:]:
                if actor_type == INSPECTOR_KIND:
                    assert from_status is not None
                    assert to_status in EVENT_STATUSES
                    assert isinstance(actor, str) and 1 <= len(actor) <= 32
                    assert isinstance(created_ms, int)
        finally:
            repo.close()
