"""Unit tests for chimera_patrol.review.state_machine (task 10.3)."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from chimera_patrol.review.state_machine import (
    EventNotFound,
    EventStateMachine,
    InvalidTransition,
    can_transition,
)
from chimera_patrol.store.repository import EVENT_STATUSES, EventRepository, NewEvent, StatusConflictError

SYSTEM_ALLOWED = [
    ("pending_review", "confirmed"),
    ("pending_review", "dismissed"),
    ("pending_review", "needs_human"),
    ("pending_review", "review_failed"),
]

INSPECTOR_ALLOWED = [
    ("needs_human", "confirmed"),
    ("needs_human", "dismissed"),
    ("review_failed", "confirmed"),
    ("review_failed", "dismissed"),
    ("confirmed", "dismissed"),
    ("dismissed", "confirmed"),
    ("confirmed", "needs_human"),
    ("dismissed", "needs_human"),
]

EVENT_ID = "evt_000000000001"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def repo(db_path: Path) -> Iterator[EventRepository]:
    repository = EventRepository(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
        "VALUES ('cam1', 'Camera 1', 'file', 'clip.mp4', 0)"
    )
    conn.commit()
    conn.close()
    yield repository
    repository.close()


def _new_event(event_id: str = EVENT_ID) -> NewEvent:
    return NewEvent(
        id=event_id,
        camera_id="cam1",
        event_type="intrusion",
        detection_source="yolo",
        start_ms=1_000,
        end_ms=1_000,
        keyframe_annotated_path="annotated.jpg",
        keyframe_raw_path="raw.jpg",
        created_ms=1_000,
    )


def _event_in_status(repo: EventRepository, status: str, event_id: str = EVENT_ID) -> None:
    """Create a pending event, then move it to ``status`` through a system transition."""
    repo.insert_event(_new_event(event_id))
    if status != "pending_review":
        EventStateMachine(repo).system_transition(event_id, status, "vlm_reviewer", now_ms=2_000)


def _status_log(db_path: Path, event_id: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT from_status, to_status, actor_type, actor, reason, created_ms "
            "FROM event_status_log WHERE event_id = ? ORDER BY id",
            (event_id,),
        ).fetchall()
    finally:
        conn.close()


def _event_status(db_path: Path, event_id: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT status FROM events WHERE id = ?", (event_id,)).fetchone()[0]
    finally:
        conn.close()


# --- can_transition: pure table checks -----------------------------------------------------


@pytest.mark.parametrize(("from_status", "to_status"), SYSTEM_ALLOWED)
def test_can_transition_allows_system_pairs(from_status: str, to_status: str) -> None:
    assert can_transition(from_status, to_status, "system") is True


@pytest.mark.parametrize(("from_status", "to_status"), INSPECTOR_ALLOWED)
def test_can_transition_allows_inspector_pairs(from_status: str, to_status: str) -> None:
    assert can_transition(from_status, to_status, "inspector") is True


@pytest.mark.parametrize(
    ("from_status", "to_status", "actor_type"),
    [
        ("pending_review", "pending_review", "system"),
        ("pending_review", "pending_review", "inspector"),
        ("confirmed", "pending_review", "inspector"),
        ("dismissed", "pending_review", "inspector"),
        ("needs_human", "pending_review", "inspector"),
        ("review_failed", "pending_review", "inspector"),
        ("pending_review", "confirmed", "inspector"),
        ("pending_review", "dismissed", "inspector"),
        ("pending_review", "needs_human", "inspector"),
        ("pending_review", "review_failed", "inspector"),
        ("review_failed", "needs_human", "inspector"),
        ("review_failed", "needs_human", "system"),
        ("confirmed", "confirmed", "inspector"),
        ("dismissed", "dismissed", "inspector"),
        ("needs_human", "needs_human", "system"),
        ("confirmed", "review_failed", "inspector"),
        ("confirmed", "needs_human", "system"),
        ("confirmed", "dismissed", "system"),
        ("needs_human", "review_failed", "system"),
        ("review_failed", "confirmed", "system"),
        ("bogus", "confirmed", "system"),
        ("pending_review", "bogus", "system"),
        ("pending_review", "confirmed", "operator"),
    ],
)
def test_can_transition_rejects_sample_pairs(
    from_status: str, to_status: str, actor_type: str
) -> None:
    assert can_transition(from_status, to_status, actor_type) is False


def test_can_transition_matches_contract_table_exhaustively() -> None:
    expected_system = set(SYSTEM_ALLOWED)
    expected_inspector = set(INSPECTOR_ALLOWED)
    for from_status in EVENT_STATUSES:
        for to_status in EVENT_STATUSES:
            pair = (from_status, to_status)
            assert can_transition(from_status, to_status, "system") is (pair in expected_system)
            assert can_transition(from_status, to_status, "inspector") is (pair in expected_inspector)


# --- EventStateMachine: allowed transitions --------------------------------------------------


@pytest.mark.parametrize(("from_status", "to_status"), SYSTEM_ALLOWED)
def test_system_transition_applies_each_allowed_pair(
    repo: EventRepository, db_path: Path, from_status: str, to_status: str
) -> None:
    _event_in_status(repo, from_status)
    EventStateMachine(repo).system_transition(EVENT_ID, to_status, "vlm_reviewer", now_ms=5_000)
    assert _event_status(db_path, EVENT_ID) == to_status


@pytest.mark.parametrize(("from_status", "to_status"), INSPECTOR_ALLOWED)
def test_override_applies_each_allowed_pair(
    repo: EventRepository, db_path: Path, from_status: str, to_status: str
) -> None:
    _event_in_status(repo, from_status)
    EventStateMachine(repo).override(EVENT_ID, to_status, "张三", now_ms=5_000)
    assert _event_status(db_path, EVENT_ID) == to_status


# --- EventStateMachine: rejected transitions -------------------------------------------------


def test_override_from_pending_review_is_rejected(repo: EventRepository, db_path: Path) -> None:
    _event_in_status(repo, "pending_review")
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).override(EVENT_ID, "confirmed", "inspector_a")
    assert _event_status(db_path, EVENT_ID) == "pending_review"
    assert len(_status_log(db_path, EVENT_ID)) == 1


def test_override_to_pending_review_is_rejected(repo: EventRepository, db_path: Path) -> None:
    _event_in_status(repo, "confirmed")
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).override(EVENT_ID, "pending_review", "inspector_a")
    assert _event_status(db_path, EVENT_ID) == "confirmed"


def test_override_to_same_state_is_rejected(repo: EventRepository, db_path: Path) -> None:
    _event_in_status(repo, "dismissed")
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).override(EVENT_ID, "dismissed", "inspector_a")
    assert len(_status_log(db_path, EVENT_ID)) == 2


def test_review_failed_to_needs_human_is_rejected(repo: EventRepository, db_path: Path) -> None:
    _event_in_status(repo, "review_failed")
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).override(EVENT_ID, "needs_human", "inspector_a")
    assert _event_status(db_path, EVENT_ID) == "review_failed"


def test_system_transition_from_non_pending_is_rejected(
    repo: EventRepository, db_path: Path
) -> None:
    _event_in_status(repo, "needs_human")
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).system_transition(EVENT_ID, "confirmed", "vlm_reviewer")
    assert _event_status(db_path, EVENT_ID) == "needs_human"


def test_unknown_event_raises_event_not_found(repo: EventRepository) -> None:
    with pytest.raises(EventNotFound) as info:
        EventStateMachine(repo).override("evt_missing", "confirmed", "inspector_a")
    assert isinstance(info.value, LookupError)


# --- Audit log and timestamps ----------------------------------------------------------------


def test_override_writes_status_log_row_with_operator_and_times(
    repo: EventRepository, db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _event_in_status(repo, "needs_human")
    caplog.set_level(logging.INFO, logger="chimera_patrol.status")

    EventStateMachine(repo).override(
        EVENT_ID, "dismissed", "张三", "误报，人员为维修工", now_ms=7_500
    )

    rows = _status_log(db_path, EVENT_ID)
    assert rows[-1] == ("needs_human", "dismissed", "inspector", "张三", "误报，人员为维修工", 7_500)
    assert _event_status(db_path, EVENT_ID) == "dismissed"

    conn = sqlite3.connect(db_path)
    updated_ms = conn.execute("SELECT updated_ms FROM events WHERE id = ?", (EVENT_ID,)).fetchone()[0]
    conn.close()
    assert updated_ms == 7_500
    assert "status change" in caplog.text and "张三" in caplog.text


def test_system_transition_writes_log_row_with_system_actor(
    repo: EventRepository, db_path: Path
) -> None:
    _event_in_status(repo, "pending_review")
    EventStateMachine(repo).system_transition(EVENT_ID, "review_failed", "startup_recovery", "重启遗留", now_ms=9_000)

    rows = _status_log(db_path, EVENT_ID)
    assert rows[0] == (None, "pending_review", "system", "cv_pipeline", None, 1_000)
    assert rows[-1] == ("pending_review", "review_failed", "system", "startup_recovery", "重启遗留", 9_000)


def test_transition_uses_current_time_when_now_ms_omitted(
    repo: EventRepository, db_path: Path
) -> None:
    _event_in_status(repo, "pending_review")
    before = int(time.time() * 1000)
    EventStateMachine(repo).system_transition(EVENT_ID, "confirmed", "vlm_reviewer")
    after = int(time.time() * 1000)

    created_ms = _status_log(db_path, EVENT_ID)[-1][-1]
    assert before <= created_ms <= after


def test_repository_transition_status_rejects_stale_from_status(
    repo: EventRepository, db_path: Path
) -> None:
    _event_in_status(repo, "confirmed")
    with pytest.raises(StatusConflictError):
        repo.transition_status(
            EVENT_ID,
            from_status="pending_review",
            to_status="dismissed",
            actor_type="system",
            actor="vlm_reviewer",
            reason=None,
            now_ms=3_000,
        )
    assert _event_status(db_path, EVENT_ID) == "confirmed"
    assert len(_status_log(db_path, EVENT_ID)) == 2


# --- Operator length (assumption A9: 1 to 32 characters) -------------------------------------


@pytest.mark.parametrize("operator", ["a", "张三", "x" * 32])
def test_operator_within_limits_is_accepted(
    repo: EventRepository, db_path: Path, operator: str
) -> None:
    _event_in_status(repo, "needs_human")
    EventStateMachine(repo).override(EVENT_ID, "confirmed", operator, now_ms=4_000)
    assert _status_log(db_path, EVENT_ID)[-1][3] == operator


@pytest.mark.parametrize("operator", ["", "x" * 33])
def test_operator_outside_limits_is_rejected_without_change(
    repo: EventRepository, db_path: Path, operator: str
) -> None:
    _event_in_status(repo, "needs_human")
    with pytest.raises(ValueError):
        EventStateMachine(repo).override(EVENT_ID, "confirmed", operator)
    assert _event_status(db_path, EVENT_ID) == "needs_human"
    assert len(_status_log(db_path, EVENT_ID)) == 2
