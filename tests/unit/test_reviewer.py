"""Unit tests for chimera_patrol.review.reviewer (task 10.5)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.vlm.normalize import VLMResponseError, VLMVerdict

EVENT_ID = "evt_000000000001"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"


class FakeAdapter:
    """Adapter stub that returns a fixed verdict, raises, or sleeps before answering."""

    def __init__(
        self,
        *,
        result: VLMVerdict | None = None,
        error: BaseException | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.result = result
        self.error = error
        self.delay_s = delay_s
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def review_event(self, req: Any) -> VLMVerdict:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            assert self.result is not None
            return self.result
        finally:
            self.in_flight -= 1


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "reviewer.db"


@pytest.fixture
def repo(db_path: Path) -> Iterator[EventRepository]:
    repository = EventRepository(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
        "VALUES ('cam1', 'Gate Camera', 'file', 'gate.mp4', 0)"
    )
    conn.commit()
    conn.close()
    yield repository
    repository.close()


def _insert_event(repo: EventRepository, event_id: str = EVENT_ID) -> None:
    repo.insert_event(
        NewEvent(
            id=event_id,
            camera_id="cam1",
            event_type="intrusion",
            detection_source="yolo",
            start_ms=1_000,
            end_ms=1_000,
            keyframe_annotated_path=f"cam1/{event_id}/keyframe_annotated.jpg",
            keyframe_raw_path=f"cam1/{event_id}/keyframe_raw.jpg",
            created_ms=1_000,
            cv_confidence=0.8,
            track_id=3,
            roi_id="gate_zone",
            roi_type="intrusion",
            cv_detail={"label": "person"},
            trajectory=[[1_000, 1.0, 2.0, 3.0, 4.0]],
        )
    )


def _event_row(db_path: Path, event_id: str = EVENT_ID) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row)


def _last_status_log(db_path: Path, event_id: str = EVENT_ID) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM event_status_log WHERE event_id = ? ORDER BY id DESC LIMIT 1",
        (event_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def _verdict(verdict: str, confidence: float) -> VLMVerdict:
    return VLMVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        description="gate area person",
        suggestion="check",
    )


def _make_reviewer(
    repo: EventRepository,
    adapter: FakeAdapter,
    *,
    timeout_s: float = 5.0,
    max_concurrency: int = 2,
    review_enabled: bool = True,
    threshold: float = 0.7,
) -> VlmReviewer:
    return VlmReviewer(
        adapter,
        EventStateMachine(repo),
        repo,
        timeout_s=timeout_s,
        max_concurrency=max_concurrency,
        description_max_chars=50,
        auto_decision_threshold=threshold,
        review_enabled=review_enabled,
    )


async def _review(reviewer: VlmReviewer, event_id: str = EVENT_ID, image: bytes | None = JPEG):
    return await reviewer.review(
        event_id,
        event_type="intrusion",
        camera_name="Gate Camera",
        roi_name="gate_zone",
        cv_confidence=0.8,
        image_jpeg=image,
    )


def _run_review(
    repo: EventRepository, adapter: FakeAdapter, **kwargs: Any
):
    async def scenario():
        reviewer = _make_reviewer(repo, adapter, **kwargs)
        return await _review(reviewer)

    return asyncio.run(scenario())


def test_true_positive_high_confidence_confirms(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("true_positive", 0.95))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "confirmed"
    assert outcome.error is None
    row = _event_row(db_path)
    assert row["status"] == "confirmed"
    assert row["vlm_verdict"] == "true_positive"
    assert row["vlm_confidence"] == pytest.approx(0.95)
    assert row["vlm_description"] == "gate area person"
    assert row["vlm_suggestion"] == "check"
    assert row["vlm_error"] is None
    assert row["vlm_reviewed_ms"] is not None
    assert _last_status_log(db_path)["actor"] == "vlm_reviewer"


def test_false_positive_high_confidence_dismisses(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("false_positive", 0.9))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "dismissed"
    assert _event_row(db_path)["status"] == "dismissed"


def test_uncertain_verdict_goes_to_needs_human(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("uncertain", 0.99))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "needs_human"
    assert _event_row(db_path)["status"] == "needs_human"


def test_low_confidence_goes_to_needs_human(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("true_positive", 0.3))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "needs_human"
    assert _event_row(db_path)["status"] == "needs_human"


def test_timeout_marks_review_failed(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("true_positive", 0.9), delay_s=1.0)

    outcome = _run_review(repo, adapter, timeout_s=0.05)

    assert outcome.status == "review_failed"
    assert outcome.error == "timeout"
    row = _event_row(db_path)
    assert row["status"] == "review_failed"
    assert row["vlm_error"] == "timeout"
    assert row["vlm_verdict"] is None
    log = _last_status_log(db_path)
    assert log["actor"] == "vlm_reviewer"
    assert log["reason"] == "timeout"


def test_adapter_exception_marks_review_failed_with_truncated_summary(
    repo: EventRepository, db_path: Path
) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(error=RuntimeError("upstream exploded " + "x" * 400))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "review_failed"
    row = _event_row(db_path)
    assert row["status"] == "review_failed"
    assert row["vlm_error"].startswith("vlm_error: RuntimeError: upstream exploded")
    assert len(row["vlm_error"]) <= 200


def test_invalid_response_marks_response_invalid(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(error=VLMResponseError("verdict must be one of ..."))

    outcome = _run_review(repo, adapter)

    assert outcome.status == "review_failed"
    assert outcome.error == "response_invalid"
    assert _event_row(db_path)["vlm_error"] == "response_invalid"


def test_missing_keyframe_marks_review_failed_without_calling_adapter(
    repo: EventRepository, db_path: Path
) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("true_positive", 0.9))

    async def scenario():
        reviewer = _make_reviewer(repo, adapter)
        return await _review(reviewer, image=None)

    outcome = asyncio.run(scenario())

    assert outcome.status == "review_failed"
    assert outcome.error == "keyframe_missing"
    assert adapter.calls == 0
    assert _event_row(db_path)["vlm_error"] == "keyframe_missing"


def test_failure_keeps_original_event_fields(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    before = _event_row(db_path)
    adapter = FakeAdapter(error=RuntimeError("boom"))

    _run_review(repo, adapter)

    after = _event_row(db_path)
    for column in (
        "event_type",
        "detection_source",
        "start_ms",
        "end_ms",
        "cv_confidence",
        "track_id",
        "roi_id",
        "roi_type",
        "cv_detail_json",
        "trajectory_json",
        "keyframe_annotated_path",
        "keyframe_raw_path",
        "created_ms",
    ):
        assert after[column] == before[column], column


def test_review_disabled_skips_adapter_and_needs_human(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("false_positive", 0.99))

    outcome = _run_review(repo, adapter, review_enabled=False)

    assert adapter.calls == 0
    assert outcome.status == "needs_human"
    assert outcome.error is None
    row = _event_row(db_path)
    assert row["status"] == "needs_human"
    assert row["review_skipped"] == 1
    assert row["vlm_verdict"] is None
    log = _last_status_log(db_path)
    assert log["actor"] == "review_switch"
    assert log["from_status"] == "pending_review"


def test_concurrency_limit_one_never_overlaps(repo: EventRepository, db_path: Path) -> None:
    for index in range(3):
        _insert_event(repo, event_id=f"evt_00000000010{index}")
    adapter = FakeAdapter(result=_verdict("true_positive", 0.9), delay_s=0.05)

    async def scenario():
        reviewer = _make_reviewer(repo, adapter, max_concurrency=1)
        return await asyncio.gather(
            *(_review(reviewer, event_id=f"evt_00000000010{i}") for i in range(3))
        )

    outcomes = asyncio.run(scenario())

    assert adapter.calls == 3
    assert adapter.max_in_flight == 1
    assert {o.status for o in outcomes} == {"confirmed"}


def test_cancellation_reraises_and_keeps_state(repo: EventRepository, db_path: Path) -> None:
    _insert_event(repo)
    adapter = FakeAdapter(result=_verdict("true_positive", 0.9), delay_s=5.0)

    async def scenario():
        reviewer = _make_reviewer(repo, adapter, timeout_s=10.0)
        task = asyncio.ensure_future(_review(reviewer))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    row = _event_row(db_path)
    assert row["status"] == "pending_review"
    assert row["vlm_error"] is None
    assert row["vlm_verdict"] is None


def test_constructor_rejects_invalid_timeout(repo: EventRepository) -> None:
    with pytest.raises(ValueError):
        VlmReviewer(
            FakeAdapter(),
            EventStateMachine(repo),
            repo,
            timeout_s=0,
            max_concurrency=1,
            description_max_chars=50,
            auto_decision_threshold=0.7,
            review_enabled=True,
        )
