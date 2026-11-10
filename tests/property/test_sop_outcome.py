"""Property 22: SOP 检查结局。

# Feature: chimera-patrol-inspection, Property 22: SOP 检查结局
**Validates: Requirements 11.6, 11.9, 11.10**

每个示例使用临时 SQLite 与真实 EventPipeline，假适配器按示例决定判定结果或失败类型。
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from chimera_patrol.config import EvidenceSection, SopSection
from chimera_patrol.evidence.ring_buffer import RingBuffer
from chimera_patrol.pipeline import EventPipeline
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.sop.checker import SopChecker, SopTriggerError
from chimera_patrol.sop.loader import SopDefinition, SopStep
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.sop_repo import SopRepository
from chimera_patrol.vlm.normalize import VLMResponseError
from chimera_patrol.vlm.protocol import SopCheckResult

STEP_IDS = ["wear_helmet", "wear_gloves", "wear_vest", "barrier_in_place"]
NOW = 1_790_000_000_000


def _jpeg() -> bytes:
    image = np.full((24, 32, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


class FakeAdapter:
    """按 mode 返回判定结果或抛出失败。"""

    def __init__(self, mode: str, fail_ids: set[str], delay_s: float) -> None:
        self._mode = mode
        self._fail_ids = fail_ids
        self._delay_s = delay_s

    async def check_sop(self, req):
        if self._mode == "timeout":
            await asyncio.sleep(self._delay_s)
        if self._mode == "error":
            raise RuntimeError("adapter boom")
        if self._mode == "invalid":
            raise VLMResponseError("not json")
        if self._mode == "bad_status":
            return SopCheckResult(steps=[(sid, "maybe", "x", None) for sid, _d, _c in req.steps])
        steps = []
        for index, (sid, _d, _c) in enumerate(req.steps):
            status = "fail" if sid in self._fail_ids else "pass"
            steps.append((sid, status, f"依据 {sid}", index))
        return SopCheckResult(steps=steps)


def _run_case(mode: str, fail_ids: set[str], with_frames: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        repo = EventRepository(base / "chimera.db")
        repo.upsert_camera("cam_a", name="东门", source_type="file", source="a.mp4", status="offline", now_ms=NOW)
        repo.set_camera_status("cam_a", "online", NOW)

        evidence = EvidenceSection(base_dir=base / "evidence")
        sop_cfg = SopSection(sample_count=2, window_before_s=1, window_after_s=0, endpoints_inclusive=True)
        ring = RingBuffer(window_s=20, max_fps=10)
        import time as _time

        now = _time.time_ns() // 1_000_000
        if with_frames:
            ring.push(now - 800, _jpeg())
            ring.push(now - 200, _jpeg())

        sop = SopDefinition(
            id="entry_prep",
            name="入场作业准备",
            camera_ids=["*"],
            steps=[SopStep(id=sid, description=f"描述 {sid}", criteria="判定标准") for sid in STEP_IDS],
        )
        pipeline = EventPipeline(repository=repo, ring_buffers={"cam_a": ring}, evidence=evidence)
        sop_repo = SopRepository(repo)
        checker = SopChecker(
            definitions=[sop],
            repository=repo,
            sop_repo=sop_repo,
            pipeline=pipeline,
            state=EventStateMachine(repo),
            adapter=FakeAdapter(mode, fail_ids, delay_s=0.3),
            ring_buffers={"cam_a": ring},
            sop_cfg=sop_cfg,
            evidence_cfg=evidence,
            timeout_s=0.05,
            semaphore=asyncio.Semaphore(1),
        )

        async def scenario() -> str:
            run_id = await checker.trigger(sop_id="entry_prep", camera_id="cam_a", operator="张三")
            for _ in range(500):
                if not checker._tasks:  # noqa: SLF001 - 测试需要等待后台检查结束
                    break
                await asyncio.sleep(0.02)
            await pipeline.close()
            return run_id

        run_id = asyncio.run(scenario())
        record = sop_repo.get_run(run_id)
        assert record is not None
        events, _ = repo.list_events(event_type="sop_violation")
        result = {
            "status": record.row["status"],
            "reason": record.row["failure_reason"],
            "steps": record.steps,
            "events": events,
            "run_id": run_id,
        }
        repo.close()
        return result


@st.composite
def case(draw):
    mode = draw(st.sampled_from(["ok", "ok", "ok", "timeout", "error", "invalid", "bad_status", "no_frames"]))
    fail_ids = draw(st.sets(st.sampled_from(STEP_IDS), max_size=len(STEP_IDS)))
    return mode, fail_ids


@settings(max_examples=100, deadline=None)
@given(data=case())
def test_sop_outcome(data) -> None:
    mode, fail_ids = data
    with_frames = mode != "no_frames"
    adapter_mode = "ok" if mode == "no_frames" else mode
    out = _run_case(adapter_mode, fail_ids, with_frames)

    if mode == "ok":
        assert out["status"] == "completed"
        assert out["reason"] is None
        assert [s["status"] for s in out["steps"]] == ["fail" if sid in fail_ids else "pass" for sid in STEP_IDS]
        if fail_ids:
            assert len(out["events"]) == 1
            event = out["events"][0]
            assert event["event_type"] == "sop_violation"
            assert event["sop_run_id"] == out["run_id"]
            assert event["status"] == "needs_human"
            failed_in_steps = {s["step_id"] for s in out["steps"] if s["status"] == "fail"}
            assert failed_in_steps == fail_ids
        else:
            assert out["events"] == []
        return

    # 任一失败类型：不产生事件，不写入任何步骤。
    expected_reason = {
        "timeout": "vlm_timeout",
        "error": "vlm_error",
        "invalid": "response_invalid",
        "bad_status": "response_invalid",
        "no_frames": "no_frames",
    }[mode]
    assert out["status"] == "check_failed_retry"
    assert out["reason"] == expected_reason
    assert out["steps"] == []
    assert out["events"] == []


def test_retry_of_must_point_to_failed_run(tmp_path: Path) -> None:
    """retry_of 只接受 check_failed_retry 的运行（仅示例校验，与属性测试互补）。"""
    repo = EventRepository(tmp_path / "chimera.db")
    repo.upsert_camera("cam_a", name="东门", source_type="file", source="a.mp4", status="online", now_ms=NOW)
    sop_repo = SopRepository(repo)
    evidence = EvidenceSection(base_dir=tmp_path / "evidence")
    sop = SopDefinition(id="entry_prep", name="x", camera_ids=["*"], steps=[SopStep(id="s", description="d", criteria="c")])
    checker = SopChecker(
        definitions=[sop],
        repository=repo,
        sop_repo=sop_repo,
        pipeline=EventPipeline(repository=repo, ring_buffers={}, evidence=evidence),
        state=EventStateMachine(repo),
        adapter=FakeAdapter("ok", set(), 0.0),
        ring_buffers={},
        sop_cfg=SopSection(sample_count=1, window_before_s=0, window_after_s=0),
        evidence_cfg=evidence,
        timeout_s=1,
        semaphore=asyncio.Semaphore(1),
    )

    async def scenario() -> None:
        try:
            await checker.trigger(sop_id="entry_prep", camera_id="cam_a", operator="张三", retry_of="sop_missing00000")
        except SopTriggerError as exc:
            assert exc.code == "not_found"
        else:
            raise AssertionError("missing retry_of must be rejected")
        await checker.close()

    asyncio.run(scenario())
    repo.close()
