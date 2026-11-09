"""Integration tests for chimera_patrol.pipeline.EventPipeline (task 6.1).

Each test drives the real EventRepository (SQLite), real RingBuffer JPEG frames,
and the real keyframe and clip writers. Coroutines run with ``asyncio.run`` so
no async pytest plugin is needed.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from chimera_patrol.config import EvidenceSection
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.pipeline import (
    STATUS_DROPPED_NO_KEYFRAME,
    STATUS_PENDING_REVIEW,
    EventPipeline,
)
from chimera_patrol.rules.types import CandidateEvent, OverlayItem
from chimera_patrol.store.repository import EventRepository

CAMERA = "cam1"


def _jpeg(value: int) -> bytes:
    frame = np.full((48, 64, 3), value % 256, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _make_repo(tmp_path: Path, cameras: tuple[str, ...] = (CAMERA,)) -> EventRepository:
    db_path = tmp_path / "chimera.db"
    repo = EventRepository(db_path)  # creates schema
    conn = sqlite3.connect(str(db_path))
    for cam in cameras:
        conn.execute(
            "INSERT INTO cameras (id, name, source_type, source, status, status_changed_ms) "
            "VALUES (?, ?, 'file', ?, 'online', 0)",
            (cam, cam, f"{cam}.mp4"),
        )
    conn.commit()
    conn.close()
    return repo


def _ring_with_frames(base_ms: int, count: int = 6, step_ms: int = 20) -> RingBuffer:
    ring = RingBuffer(window_s=600, max_fps=1000)
    for i in range(count):
        assert ring.push(base_ms + i * step_ms, _jpeg(i * 10))
    return ring


def _candidate(base_ms: int, camera_id: str = CAMERA) -> CandidateEvent:
    return CandidateEvent(
        camera_id=camera_id,
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


def _evidence(tmp_path: Path, post_s: float = 0.05) -> EvidenceSection:
    return EvidenceSection(
        base_dir=tmp_path / "evidence",
        clip_pre_s=0,
        clip_post_s=post_s,
        clip_fps=10,
    )


async def _wait_until(predicate: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.02)


def _status_log_rows(db_path: Path, event_id: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT from_status, to_status, actor_type FROM event_status_log "
            "WHERE event_id = ? ORDER BY id",
            (event_id,),
        ).fetchall()
    finally:
        conn.close()


def test_submit_persists_event_with_keyframes_and_pending_status(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    evidence = _evidence(tmp_path, post_s=60)  # clip never completes during this test
    ring = _ring_with_frames(base)
    created: list[dict[str, Any]] = []
    reviewed: list[str] = []

    async def scenario() -> str:
        async def on_created(summary: dict[str, Any]) -> None:
            created.append(summary)

        async def on_review(event_id: str) -> None:
            reviewed.append(event_id)

        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: ring},
            evidence=evidence,
            on_created=on_created,
            on_review=on_review,
        )
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        assert result.status == STATUS_PENDING_REVIEW
        assert result.detail is None
        return result.event_id

    event_id = asyncio.run(scenario())

    assert event_id.startswith("evt_") and len(event_id) == len("evt_") + 12
    event = repo.get_event(event_id)
    assert event is not None
    assert event["status"] == "pending_review"
    assert event["clip_status"] == "pending"
    assert event["trajectory"] == [[base, 10.0, 10.0, 50.0, 80.0]]
    assert event["cv_detail"] == {"label": "person", "box": [10, 10, 50, 80]}

    annotated = evidence.base_dir / event["keyframe_annotated_path"]
    raw = evidence.base_dir / event["keyframe_raw_path"]
    assert annotated.is_file() and annotated.stat().st_size > 0
    assert raw.is_file() and raw.stat().st_size > 0

    rows = _status_log_rows(tmp_path / "chimera.db", event_id)
    assert rows == [(None, "pending_review", "system")]

    assert created == [
        {
            "id": event_id,
            "camera_id": CAMERA,
            "event_type": "intrusion",
            "status": "pending_review",
        }
    ]
    assert reviewed == [event_id]


def test_no_keyframe_drops_event_and_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    created: list[dict[str, Any]] = []

    async def on_created(summary: dict[str, Any]) -> None:
        created.append(summary)

    async def scenario() -> Any:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: RingBuffer(window_s=60, max_fps=10)},  # empty buffer
            evidence=_evidence(tmp_path),
            on_created=on_created,
        )
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        return result

    with caplog.at_level(logging.ERROR, logger="chimera_patrol.pipeline"):
        result = asyncio.run(scenario())

    assert result.status == STATUS_DROPPED_NO_KEYFRAME
    assert result.detail
    assert repo.list_events()[1] == 0
    assert repo.get_event(result.event_id) is None
    assert created == []
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(CAMERA in r.getMessage() and result.event_id in r.getMessage() for r in error_records)


def test_camera_without_ring_buffer_is_dropped(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)

    async def scenario() -> Any:
        pipeline = EventPipeline(repository=repo, ring_buffers={}, evidence=_evidence(tmp_path))
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        return result

    result = asyncio.run(scenario())
    assert result.status == STATUS_DROPPED_NO_KEYFRAME
    assert repo.list_events()[1] == 0


def test_insert_failure_is_reraised_to_caller(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path, cameras=())  # camera row missing -> FK violation on insert
    ring = _ring_with_frames(base)

    async def scenario() -> None:
        pipeline = EventPipeline(
            repository=repo, ring_buffers={CAMERA: ring}, evidence=_evidence(tmp_path)
        )
        try:
            await pipeline.submit(_candidate(base))
        finally:
            await pipeline.close()

    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(scenario())
    assert repo.list_events()[1] == 0


def test_clip_task_marks_clip_ready(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    evidence = _evidence(tmp_path, post_s=0.05)
    ring = _ring_with_frames(base)
    clip_done: list[tuple[str, str]] = []

    async def on_clip_done(event_id: str, status: str) -> None:
        clip_done.append((event_id, status))

    async def scenario() -> str:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: ring},
            evidence=evidence,
            on_clip_done=on_clip_done,
        )
        result = await pipeline.submit(_candidate(base))
        await _wait_until(lambda: len(clip_done) == 1)
        await pipeline.close()
        return result.event_id

    event_id = asyncio.run(scenario())

    event = repo.get_event(event_id)
    assert event is not None
    assert event["clip_status"] == "ready"
    assert event["clip_path"] == f"{CAMERA}/{event_id}/clip.mp4"
    assert (evidence.base_dir / event["clip_path"]).is_file()
    assert event["status"] == "pending_review"
    assert clip_done == [(event_id, "ready")]


def test_clip_failure_does_not_change_event_status(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    ring = _ring_with_frames(base)
    clip_done: list[tuple[str, str]] = []
    reviewed: list[str] = []

    def failing_saver(*_: Any, **__: Any) -> tuple[str, str]:
        raise RuntimeError("encoder exploded")

    async def on_clip_done(event_id: str, status: str) -> None:
        clip_done.append((event_id, status))

    async def on_review(event_id: str) -> None:
        reviewed.append(event_id)

    async def scenario() -> str:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: ring},
            evidence=_evidence(tmp_path, post_s=0.05),
            on_clip_done=on_clip_done,
            on_review=on_review,
            clip_saver=failing_saver,
        )
        result = await pipeline.submit(_candidate(base))
        await _wait_until(lambda: len(clip_done) == 1)
        await pipeline.close()
        return result.event_id

    event_id = asyncio.run(scenario())

    event = repo.get_event(event_id)
    assert event is not None
    assert event["clip_status"] == "failed"
    assert event["clip_path"] is None
    assert event["status"] == "pending_review"
    assert clip_done == [(event_id, "failed")]
    assert reviewed == [event_id]


def test_clip_job_is_cancelled_on_close(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    ring = _ring_with_frames(base)
    clip_done: list[tuple[str, str]] = []

    async def on_clip_done(event_id: str, status: str) -> None:
        clip_done.append((event_id, status))

    async def scenario() -> str:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: ring},
            evidence=_evidence(tmp_path, post_s=60),
            on_clip_done=on_clip_done,
        )
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        with pytest.raises(RuntimeError):
            await pipeline.submit(_candidate(base))
        return result.event_id

    event_id = asyncio.run(scenario())

    assert clip_done == []
    event = repo.get_event(event_id)
    assert event is not None
    assert event["clip_status"] == "pending"


def test_hook_failure_does_not_remove_event(tmp_path: Path) -> None:
    base = int(time.time() * 1000)
    repo = _make_repo(tmp_path)
    ring = _ring_with_frames(base)

    async def broken_created(_: dict[str, Any]) -> None:
        raise ValueError("broadcast failed")

    async def scenario() -> str:
        pipeline = EventPipeline(
            repository=repo,
            ring_buffers={CAMERA: ring},
            evidence=_evidence(tmp_path, post_s=60),
            on_created=broken_created,
        )
        result = await pipeline.submit(_candidate(base))
        await pipeline.close()
        return result.event_id

    event_id = asyncio.run(scenario())
    event = repo.get_event(event_id)
    assert event is not None and event["status"] == "pending_review"
