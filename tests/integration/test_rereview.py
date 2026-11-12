"""VLM 复检：状态机、复核器与 API 的测试。

- 只允许 needs_human、review_failed 发起复检，迁移为 pending_review，并写入留痕日志。
- 复检开始时清空上一次的 VLM 字段；复核结论仍按原决策规则迁移。
- 复核关闭时禁止复检。
- API：202 与后台调用；非法状态 409；复核关闭 422；未知事件 404。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chimera_patrol.api.app import create_app
from chimera_patrol.api.context import AppContext
from chimera_patrol.api.ws import WsHub
from chimera_patrol.config import AppConfig
from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.review.state_machine import (
    EventNotFound,
    EventStateMachine,
    InvalidTransition,
    ReviewDisabled,
)
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.vlm.mock import MockAdapter

NOW = 1_790_000_000_000
JPEG = b"\xff\xd8\xff\xe0test-jpeg-bytes"


def _repo(tmp_path: Path) -> EventRepository:
    repo = EventRepository(tmp_path / "chimera.db")
    repo.upsert_camera("cam_a", name="东门", source_type="file", source="a.mp4", status="online", now_ms=NOW)
    return repo


def _insert(repo: EventRepository, event_id: str, status: str) -> None:
    """插入事件并迁移到目标状态（pending_review 之外的状态需经系统迁移）。"""
    repo.insert_event(
        NewEvent(
            id=event_id,
            camera_id="cam_a",
            event_type="intrusion",
            detection_source="yolo",
            start_ms=NOW,
            end_ms=NOW,
            keyframe_annotated_path=f"cam_a/{event_id}/keyframe_annotated.jpg",
            keyframe_raw_path=f"cam_a/{event_id}/keyframe_raw.jpg",
            created_ms=NOW,
            cv_confidence=0.8,
        )
    )
    if status == "pending_review":
        return
    state = EventStateMachine(repo)
    if status == "needs_human":
        state.system_transition(event_id, "needs_human", "vlm_reviewer", now_ms=NOW + 1)
    elif status == "review_failed":
        state.system_transition(event_id, "review_failed", "vlm_reviewer", now_ms=NOW + 1)
    elif status == "confirmed":
        state.system_transition(event_id, "confirmed", "vlm_reviewer", now_ms=NOW + 1)
    elif status == "dismissed":
        state.system_transition(event_id, "dismissed", "vlm_reviewer", now_ms=NOW + 1)


# ---------------------------------------------------------------- 状态机


def test_rereview_from_needs_human_and_review_failed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    for event_id, status in (("evt_nh000000001", "needs_human"), ("evt_rf000000001", "review_failed")):
        _insert(repo, event_id, status)
        EventStateMachine(repo).begin_rereview(event_id, "张三", review_enabled=True, now_ms=NOW + 5)

        event = repo.get_event(event_id)
        assert event["status"] == "pending_review"
        assert event["vlm_verdict"] is None and event["vlm_error"] is None  # 旧结果已清空
        last = repo.list_status_log(event_id)[-1]
        assert (last["from_status"], last["to_status"]) == (status, "pending_review")
        assert last["actor_type"] == "inspector"
        assert last["actor"] == "张三"
        assert last["reason"] == "复检请求"
    repo.close()


@pytest.mark.parametrize("status", ["confirmed", "dismissed", "pending_review"])
def test_rereview_rejected_for_other_statuses(tmp_path: Path, status: str) -> None:
    repo = _repo(tmp_path)
    event_id = "evt_xx0000000001"
    _insert(repo, event_id, status)
    before = repo.get_event(event_id)["status"]
    with pytest.raises(InvalidTransition):
        EventStateMachine(repo).begin_rereview(event_id, "张三", review_enabled=True)
    assert repo.get_event(event_id)["status"] == before  # 不改变状态
    repo.close()


def test_rereview_rejected_when_review_disabled(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _insert(repo, "evt_off00000001", "needs_human")
    with pytest.raises(ReviewDisabled):
        EventStateMachine(repo).begin_rereview("evt_off00000001", "张三", review_enabled=False)
    assert repo.get_event("evt_off00000001")["status"] == "needs_human"
    repo.close()


def test_rereview_unknown_event(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(EventNotFound):
        EventStateMachine(repo).begin_rereview("evt_missing0001", "张三", review_enabled=True)
    repo.close()


def test_rereview_rejects_invalid_operator(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _insert(repo, "evt_op000000001", "needs_human")
    with pytest.raises(ValueError):
        EventStateMachine(repo).begin_rereview("evt_op000000001", "", review_enabled=True)
    assert repo.get_event("evt_op000000001")["status"] == "needs_human"
    repo.close()


# ---------------------------------------------------------------- 复核结论


def test_rereview_result_follows_decision_rules(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    event_id = "evt_rv000000001"
    _insert(repo, event_id, "review_failed")
    state = EventStateMachine(repo)
    state.begin_rereview(event_id, "张三", review_enabled=True)

    reviewer = VlmReviewer(
        MockAdapter(default_verdict="uncertain", by_camera={"东门": "true_positive"}),
        state,
        repo,
        timeout_s=5,
        max_concurrency=2,
        description_max_chars=50,
        auto_decision_threshold=0.7,
        review_enabled=True,
    )
    asyncio.run(
        reviewer.review(
            event_id,
            event_type="intrusion",
            camera_name="东门",
            roi_name=None,
            cv_confidence=0.8,
            image_jpeg=JPEG,
        )
    )
    event = repo.get_event(event_id)
    assert event["status"] == "confirmed"  # MockAdapter 对东门返回 true_positive，置信度 0.9 >= 0.7
    assert event["vlm_verdict"] == "true_positive"
    assert [entry["to_status"] for entry in repo.list_status_log(event_id)][-3:] == [
        "review_failed",
        "pending_review",
        "confirmed",
    ]
    repo.close()


# ---------------------------------------------------------------- API


class _RecordingRereview:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, event_id: str) -> None:
        self.calls.append(event_id)


def _client(tmp_path: Path, *, review_enabled: bool, recorder: _RecordingRereview | None):
    repo = _repo(tmp_path)
    cfg = AppConfig.model_validate(
        {
            "evidence": {"base_dir": str(tmp_path / "evidence")},
            "vlm": {"review_enabled": review_enabled, "provider": "mock"},
        }
    )
    ctx = AppContext(config=cfg, repo=repo, hub=WsHub(heartbeat_interval_s=15, vlm_review_enabled=review_enabled))
    ctx.rereview = recorder
    return TestClient(create_app(ctx)), repo


def test_api_rereview_accepts_and_schedules(tmp_path: Path) -> None:
    recorder = _RecordingRereview()
    client, repo = _client(tmp_path, review_enabled=True, recorder=recorder)
    _insert(repo, "evt_api00000001", "needs_human")

    resp = client.post("/api/events/evt_api00000001/review", json={"operator": "张三"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending_review"
    assert body["vlm"] is None
    # 后台任务由 TestClient 的事件循环执行，等待其完成后确认被调用。
    for _ in range(50):
        if recorder.calls:
            break
        time.sleep(0.02)
    assert recorder.calls == ["evt_api00000001"]
    repo.close()


def test_api_rereview_rejects_confirmed(tmp_path: Path) -> None:
    recorder = _RecordingRereview()
    client, repo = _client(tmp_path, review_enabled=True, recorder=recorder)
    _insert(repo, "evt_api00000002", "confirmed")

    resp = client.post("/api/events/evt_api00000002/review", json={"operator": "张三"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "invalid_transition"
    assert recorder.calls == []
    repo.close()


def test_api_rereview_rejects_when_review_disabled(tmp_path: Path) -> None:
    recorder = _RecordingRereview()
    client, repo = _client(tmp_path, review_enabled=False, recorder=recorder)
    _insert(repo, "evt_api00000003", "needs_human")

    resp = client.post("/api/events/evt_api00000003/review", json={"operator": "张三"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert repo.get_event("evt_api00000003")["status"] == "needs_human"
    repo.close()


def test_api_rereview_unknown_event_and_bad_operator(tmp_path: Path) -> None:
    recorder = _RecordingRereview()
    client, repo = _client(tmp_path, review_enabled=True, recorder=recorder)
    assert client.post("/api/events/evt_nothere0001/review", json={"operator": "张三"}).status_code == 404
    _insert(repo, "evt_api00000004", "needs_human")
    assert client.post("/api/events/evt_api00000004/review", json={"operator": ""}).status_code == 422
    repo.close()


def test_api_rereview_unavailable_without_runtime(tmp_path: Path) -> None:
    client, repo = _client(tmp_path, review_enabled=True, recorder=None)
    _insert(repo, "evt_api00000005", "needs_human")
    resp = client.post("/api/events/evt_api00000005/review", json={"operator": "张三"})
    assert resp.status_code == 500
    assert repo.get_event("evt_api00000005")["status"] == "needs_human"  # 未迁移
    repo.close()
