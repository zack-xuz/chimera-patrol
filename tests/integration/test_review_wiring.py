"""Integration tests for chimera_patrol.review.wiring (task 10.8).

Real EventRepository (SQLite), real EventPipeline, real RingBuffer JPEG frames and
keyframe writer. The VLM side is a MockAdapter wrapped in a call-counting spy.
Coroutines run with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chimera_patrol.config import EvidenceSection
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.review.wiring import build_on_review
from chimera_patrol.rules.types import CandidateEvent, OverlayItem
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.vlm.mock import MockAdapter
from chimera_patrol.vlm.normalize import VLMVerdict
from chimera_patrol.vlm.protocol import ReviewRequest

CAMERA = "cam1"
CAMERA_NAME = "东门通道"


class SpyAdapter:
    """Records every review call, then delegates to a MockAdapter."""

    def __init__(self, inner: MockAdapter) -> None:
        self._inner = inner
        self.requests: list[ReviewRequest] = []

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        self.requests.append(req)
        return await self._inner.review_event(req)


def _jpeg(value: int) -> bytes:
    frame = np.full((48, 64, 3), value % 256, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _make_repo(tmp_path: Path) -> EventRepository:
    db_path = tmp_path / "chimera.db"
    repo = EventRepository(db_path)  # creates schema
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO cameras (id, name, source_type, source, status, status_changed_ms) "
        "VALUES (?, ?, 'file', 'gate.mp4', 'online', 0)",
        (CAMERA, CAMERA_NAME),
    )
    conn.commit()
    conn.close()
    return repo


def _ring(base_ms: int) -> RingBuffer:
    ring = RingBuffer(window_s=600, max_fps=1000)
    for i in range(6):
        assert ring.push(base_ms + i * 20, _jpeg(i * 10))
    return ring


def _candidate(base_ms: int) -> CandidateEvent:
    return CandidateEvent(
        camera_id=CAMERA,
        event_type="intrusion",
        detection_source="yolo",
        start_ms=base_ms,
        end_ms=base_ms,
        cv_confidence=0.9,
        track_id=3,
        roi_id="gate",
        roi_type="intrusion",
        detail={"label": "person", "box": [10, 10, 50, 80]},
        trajectory=[[base_ms, 10.0, 10.0, 50.0, 80.0]],
        overlay=[
            OverlayItem(
                box=(10, 10, 50, 80),
                label="person",
                conf=0.9,
                roi_polygon=[(0, 0), (60, 0), (60, 45)],
            )
        ],
    )


def _evidence(tmp_path: Path) -> EvidenceSection:
    # clip_post_s is large so the background clip job never finishes during a test.
    return EvidenceSection(base_dir=tmp_path / "evidence", clip_pre_s=0, clip_post_s=60, clip_fps=10)


def _reviewer(
    repo: EventRepository, adapter: SpyAdapter, *, review_enabled: bool
) -> VlmReviewer:
    return VlmReviewer(
        adapter,
        EventStateMachine(repo),
        repo,
        timeout_s=5.0,
        max_concurrency=2,
        description_max_chars=50,
        auto_decision_threshold=0.7,
        review_enabled=review_enabled,
    )


def _status_log(db_path: Path, event_id: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT from_status, to_status, actor_type, actor FROM event_status_log "
            "WHERE event_id = ? ORDER BY id",
            (event_id,),
        ).fetchall()
    finally:
        conn.close()


def _submit(
    repo: EventRepository,
    tmp_path: Path,
    reviewer: VlmReviewer,
    *,
    on_created: Any = None,
) -> str:
    base = int(time.time() * 1000)
    evidence = _evidence(tmp_path)

    async def scenario() -> str:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: _ring(base)},
            evidence=evidence,
            on_created=on_created,
            on_review=build_on_review(
                repository=repo,
                reviewer=reviewer,
                base_dir=evidence.base_dir,
                camera_names={CAMERA: CAMERA_NAME},
            ),
        )
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        return result.event_id

    return asyncio.run(scenario())


def test_review_enabled_true_positive_confirms_event(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    adapter = SpyAdapter(MockAdapter(default_verdict="true_positive"))

    event_id = _submit(repo, tmp_path, _reviewer(repo, adapter, review_enabled=True))

    event = repo.get_event(event_id)
    assert event is not None
    assert event["status"] == "confirmed"
    assert event["vlm_verdict"] == "true_positive"
    assert event["vlm_confidence"] == 0.9
    assert event["review_skipped"] == 0
    assert event["vlm_error"] is None

    # The reviewer received the annotated key frame, the camera display name and the ROI id.
    assert len(adapter.requests) == 1
    req = adapter.requests[0]
    assert req.camera_name == CAMERA_NAME
    assert req.roi_name == "gate"
    annotated = (tmp_path / "evidence" / event["keyframe_annotated_path"]).read_bytes()
    assert req.image_jpeg == annotated

    log = _status_log(tmp_path / "chimera.db", event_id)
    assert log == [
        (None, "pending_review", "system", "cv_pipeline"),
        ("pending_review", "confirmed", "system", "vlm_reviewer"),
    ]


def test_review_disabled_moves_to_needs_human_without_calling_adapter(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    adapter = SpyAdapter(MockAdapter(default_verdict="true_positive"))

    event_id = _submit(repo, tmp_path, _reviewer(repo, adapter, review_enabled=False))

    assert adapter.requests == []
    event = repo.get_event(event_id)
    assert event is not None
    assert event["status"] == "needs_human"
    assert event["review_skipped"] == 1
    assert event["vlm_verdict"] is None
    assert event["vlm_error"] is None

    log = _status_log(tmp_path / "chimera.db", event_id)
    assert log[-1] == ("pending_review", "needs_human", "system", "review_switch")


def test_missing_keyframe_is_reported_as_keyframe_missing(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    adapter = SpyAdapter(MockAdapter(default_verdict="true_positive"))

    async def delete_annotated(summary: dict[str, Any]) -> None:
        # on_created runs after the key frame is saved and before on_review reads it.
        event = repo.get_event(summary["id"])
        assert event is not None
        (tmp_path / "evidence" / event["keyframe_annotated_path"]).unlink()

    event_id = _submit(
        repo,
        tmp_path,
        _reviewer(repo, adapter, review_enabled=True),
        on_created=delete_annotated,
    )

    assert adapter.requests == []
    event = repo.get_event(event_id)
    assert event is not None
    assert event["status"] == "review_failed"
    assert event["vlm_error"] == "keyframe_missing"
