"""Unit tests for chimera_patrol.review.recovery (task 10.7, assumption A10)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chimera_patrol.review.recovery import recover_pending_reviews
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.store.repository import EventRepository, NewEvent

CAMERA_ID = "cam_a"
CREATED_MS = 1790000000000


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "patrol.db"
    EventRepository(path).close()  # Creates the schema.
    # The repository has no camera writer yet, so insert the FK parent directly.
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO cameras (id, name, source_type, source, status, status_changed_ms) "
            "VALUES (?, ?, 'file', ?, 'online', ?)",
            (CAMERA_ID, "Camera A", "a.mp4", CREATED_MS),
        )
    return path


@pytest.fixture
def repo(db_path: Path) -> Iterator[EventRepository]:
    repository = EventRepository(db_path)
    yield repository
    repository.close()


def _new_event(event_id: str, offset_ms: int = 0) -> NewEvent:
    return NewEvent(
        id=event_id,
        camera_id=CAMERA_ID,
        event_type="intrusion",
        detection_source="yolo",
        start_ms=CREATED_MS + offset_ms,
        end_ms=CREATED_MS + offset_ms + 1000,
        keyframe_annotated_path=f"{CAMERA_ID}/{event_id}/keyframe_annotated.jpg",
        keyframe_raw_path=f"{CAMERA_ID}/{event_id}/keyframe_raw.jpg",
        created_ms=CREATED_MS + offset_ms,
        cv_confidence=0.9,
        track_id=3,
        roi_id="gate",
        roi_type="intrusion",
        cv_detail={"label": "person"},
        trajectory=[[CREATED_MS, 1, 2, 3, 4]],
    )


def _move_to(repo: EventRepository, event_id: str, target: str) -> None:
    """Move a pending_review event to ``target`` through a valid system transition."""
    EventStateMachine(repo).system_transition(event_id, target, actor="vlm_reviewer")


def _status_log(db_path: Path, event_id: str) -> list[sqlite3.Row]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM event_status_log WHERE event_id = ? ORDER BY id", (event_id,)
        ).fetchall()


def _event(repo: EventRepository, event_id: str) -> dict[str, Any]:
    event = repo.get_event(event_id)
    assert event is not None
    return event


def test_recovers_only_pending_review_events(repo: EventRepository, db_path: Path) -> None:
    repo.insert_event(_new_event("evt_pending000001"))
    repo.insert_event(_new_event("evt_confirmed0001", offset_ms=1))
    repo.insert_event(_new_event("evt_needhuman001", offset_ms=2))
    _move_to(repo, "evt_confirmed0001", "confirmed")
    _move_to(repo, "evt_needhuman001", "needs_human")

    before_pending = _event(repo, "evt_pending000001")
    before_confirmed = _event(repo, "evt_confirmed0001")
    before_needs = _event(repo, "evt_needhuman001")
    log_confirmed_before = len(_status_log(db_path, "evt_confirmed0001"))
    log_needs_before = len(_status_log(db_path, "evt_needhuman001"))

    recovered = recover_pending_reviews(repo, EventStateMachine(repo), now_ms=CREATED_MS + 5000)

    assert recovered == 1

    after = _event(repo, "evt_pending000001")
    assert after["status"] == "review_failed"
    assert after["vlm_error"] == "service_restart"
    # Every other column stays as it was. updated_ms moves with the status change.
    for key in set(before_pending) - {"status", "vlm_error", "updated_ms"}:
        assert after[key] == before_pending[key], key
    assert after["updated_ms"] == CREATED_MS + 5000

    last_log = _status_log(db_path, "evt_pending000001")[-1]
    assert last_log["from_status"] == "pending_review"
    assert last_log["to_status"] == "review_failed"
    assert last_log["actor_type"] == "system"
    assert last_log["actor"] == "startup_recovery"
    assert last_log["reason"] == "service_restart"
    assert last_log["created_ms"] == CREATED_MS + 5000

    assert _event(repo, "evt_confirmed0001") == before_confirmed
    assert _event(repo, "evt_needhuman001") == before_needs
    assert len(_status_log(db_path, "evt_confirmed0001")) == log_confirmed_before
    assert len(_status_log(db_path, "evt_needhuman001")) == log_needs_before


def test_second_run_recovers_nothing(repo: EventRepository, db_path: Path) -> None:
    repo.insert_event(_new_event("evt_pending000002"))
    machine = EventStateMachine(repo)

    assert recover_pending_reviews(repo, machine) == 1
    snapshot = _event(repo, "evt_pending000002")
    log_count = len(_status_log(db_path, "evt_pending000002"))

    assert recover_pending_reviews(repo, machine) == 0
    assert _event(repo, "evt_pending000002") == snapshot
    assert len(_status_log(db_path, "evt_pending000002")) == log_count


def test_empty_store_returns_zero(repo: EventRepository) -> None:
    assert recover_pending_reviews(repo, EventStateMachine(repo)) == 0


def test_recovers_more_than_one_page_of_pending_events(repo: EventRepository) -> None:
    total = 205  # Larger than the repository page limit of 200.
    for index in range(total):
        repo.insert_event(_new_event(f"evt_bulk{index:08d}", offset_ms=index))

    assert recover_pending_reviews(repo, EventStateMachine(repo)) == total

    _, remaining = repo.list_events(status="pending_review")
    _, failed = repo.list_events(status="review_failed")
    assert remaining == 0
    assert failed == total
