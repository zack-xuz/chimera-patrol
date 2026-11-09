"""Property 13: 复核失败与复核关闭（任务 10.6）。

**Validates: Requirements 5.5, 5.6**

对任意失败原因，复核失败的事件都进入 review_failed 且记录 vlm_error；
复核关闭时适配器一次都不被调用，事件进入 needs_human 且 review_skipped 为真。
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.vlm.normalize import VLMResponseError, VLMVerdict

EVENT_ID = "evt_00000000abcd"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


class _FailingAdapter:
    """每次调用都按 kind 失败，并记录调用次数。"""

    def __init__(self, kind: str, message: str) -> None:
        self._kind = kind
        self._message = message
        self.calls = 0

    async def review_event(self, req: Any) -> VLMVerdict:
        self.calls += 1
        if self._kind == "timeout":
            await asyncio.sleep(1.0)
        if self._kind == "response_invalid":
            raise VLMResponseError("bad json")
        if self._kind == "vlm_error":
            raise RuntimeError(self._message)
        raise AssertionError("unreachable: success path is not part of this property")

    async def check_sop(self, req: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    async def check_helmet_crop(self, req: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError


def _setup_repo(db_path: Path) -> EventRepository:
    repo = EventRepository(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
        "VALUES ('cam1', 'Gate', 'file', 'gate.mp4', 0)"
    )
    conn.commit()
    conn.close()
    repo.insert_event(
        NewEvent(
            id=EVENT_ID,
            camera_id="cam1",
            event_type="intrusion",
            detection_source="yolo",
            start_ms=1_000,
            end_ms=1_000,
            keyframe_annotated_path=f"cam1/{EVENT_ID}/keyframe_annotated.jpg",
            keyframe_raw_path=f"cam1/{EVENT_ID}/keyframe_raw.jpg",
            created_ms=1_000,
            cv_confidence=0.8,
            track_id=1,
            roi_id="zone",
            roi_type="intrusion",
            cv_detail={"label": "person"},
            trajectory=[],
        )
    )
    return repo


def _run_case(
    kind: str,
    review_enabled: bool,
    message: str,
    image: bytes | None,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "review.db"
        repo = _setup_repo(db_path)
        try:
            adapter = _FailingAdapter(kind, message)

            async def scenario() -> None:
                reviewer = VlmReviewer(
                    adapter,
                    EventStateMachine(repo),
                    repo,
                    timeout_s=0.01,
                    max_concurrency=2,
                    description_max_chars=50,
                    auto_decision_threshold=0.7,
                    review_enabled=review_enabled,
                )
                await reviewer.review(
                    EVENT_ID,
                    event_type="intrusion",
                    camera_name="Gate",
                    roi_name="zone",
                    cv_confidence=0.8,
                    image_jpeg=image,
                )

            asyncio.run(scenario())
            event = repo.get_event(EVENT_ID)
            assert event is not None
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            last = dict(
                conn.execute(
                    "SELECT * FROM event_status_log WHERE event_id = ? ORDER BY id DESC LIMIT 1",
                    (EVENT_ID,),
                ).fetchone()
            )
            conn.close()
            return event, adapter.calls, last
        finally:
            repo.close()


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["timeout", "response_invalid", "vlm_error"]),
    message=st.text(max_size=300),
)
def test_failed_review_is_recorded_as_review_failed(kind: str, message: str) -> None:
    # Feature: chimera-patrol-inspection, Property 13: 复核失败与复核关闭
    # **Validates: Requirements 5.5, 5.6**
    event, calls, last = _run_case(kind, True, message, JPEG)

    assert calls == 1
    assert event["status"] == "review_failed"
    assert last["to_status"] == "review_failed"
    assert last["actor_type"] == "system"
    assert last["actor"] == "vlm_reviewer"
    error = event["vlm_error"]
    assert error is not None and error != ""
    if kind == "timeout":
        assert error == "timeout"
    elif kind == "response_invalid":
        assert error == "response_invalid"
    else:
        assert error.startswith("vlm_error: ")
        assert len(error) <= 200
        assert "\n" not in error and "\r" not in error


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(message=st.text(max_size=40))
def test_missing_keyframe_fails_without_calling_adapter(message: str) -> None:
    # Feature: chimera-patrol-inspection, Property 13: 复核失败与复核关闭
    # **Validates: Requirements 5.5, 5.6**
    event, calls, last = _run_case("vlm_error", True, message, None)

    assert calls == 0
    assert event["status"] == "review_failed"
    assert event["vlm_error"] == "keyframe_missing"
    assert last["actor"] == "vlm_reviewer"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["timeout", "response_invalid", "vlm_error"]),
    message=st.text(max_size=40),
)
def test_review_disabled_never_calls_adapter(kind: str, message: str) -> None:
    # Feature: chimera-patrol-inspection, Property 13: 复核失败与复核关闭
    # **Validates: Requirements 5.5, 5.6**
    event, calls, last = _run_case(kind, False, message, JPEG)

    assert calls == 0
    assert event["status"] == "needs_human"
    assert event["review_skipped"] == 1
    assert event["vlm_error"] is None
    assert last["to_status"] == "needs_human"
    assert last["actor"] == "review_switch"
