"""API 与 WebSocket 集成测试（任务 13.6）。

使用 FastAPI TestClient 与临时 SQLite 文件。覆盖：事件列表过滤与分页、详情字段、
改判的 200 / 404 / 409 / 422、统一错误格式、MJPEG 响应头与首个分片、证据路径越界、
WebSocket 的 hello、ping/pong 与状态变化推送、SOP 触发的错误码。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chimera_patrol.api.app import create_app
from chimera_patrol.api.context import AppContext
from chimera_patrol.api.ws import WsHub
from chimera_patrol.config import AppConfig
from chimera_patrol.ingest.capture import FrameSlot
from chimera_patrol.sop.checker import SopChecker
from chimera_patrol.sop.loader import SopDefinition, SopStep
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.vlm.mock import MockAdapter
from chimera_patrol.evidence.ring_buffer import RingBuffer
import asyncio

NOW = 1_790_000_000_000


class _UnusedPipeline:
    """trigger 不调用流水线；仅在判定完成后才会使用。测试中判定立即失败，不会触达此对象。"""

    async def submit(self, *args, **kwargs):  # pragma: no cover - 不应被调用
        raise AssertionError("pipeline must not be used in these tests")


def _config(base_dir: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "evidence": {"base_dir": str(base_dir)},
            "sop": {"sample_count": 1, "window_before_s": 0, "window_after_s": 0},
        }
    )


def _event(event_id: str, camera_id: str, *, event_type: str = "intrusion", created_ms: int, status_hint: str = "pending") -> NewEvent:
    return NewEvent(
        id=event_id,
        camera_id=camera_id,
        event_type=event_type,
        detection_source="yolo",
        start_ms=created_ms,
        end_ms=created_ms,
        keyframe_annotated_path=f"{camera_id}/{event_id}/keyframe_annotated.jpg",
        keyframe_raw_path=f"{camera_id}/{event_id}/keyframe_raw.jpg",
        created_ms=created_ms,
        cv_confidence=0.8,
        track_id=1,
        roi_id="zone",
        roi_type="intrusion",
        cv_detail={"label": "person", "box": [0, 0, 10, 10]},
        trajectory=[[created_ms, 0.0, 0.0, 10.0, 10.0]],
    )


@pytest.fixture()
def env(tmp_path: Path):
    repo = EventRepository(tmp_path / "chimera.db")
    now = NOW
    repo.upsert_camera("cam_a", name="东门", source_type="file", source="a.mp4", status="offline", now_ms=now)
    repo.upsert_camera("cam_b", name="西门", source_type="file", source="b.mp4", status="offline", now_ms=now)
    repo.set_camera_status("cam_a", "online", now)

    repo.insert_event(_event("evt_000000000001", "cam_a", created_ms=now + 1))
    repo.insert_event(_event("evt_000000000002", "cam_a", created_ms=now + 2))
    repo.insert_event(_event("evt_000000000003", "cam_b", created_ms=now + 3))
    repo.insert_event(_event("evt_000000000004", "cam_b", event_type="smoke_fire", created_ms=now + 4))
    # 改判只允许从 needs_human 等状态发起，因此把 evt_000000000002 先迁到 needs_human。
    repo.transition_status(
        "evt_000000000002",
        from_status="pending_review",
        to_status="needs_human",
        actor_type="system",
        actor="vlm_reviewer",
        reason=None,
        now_ms=now + 5,
    )

    base = tmp_path / "evidence"
    cfg = _config(base)
    hub = WsHub(heartbeat_interval_s=cfg.ws.heartbeat_interval_s, vlm_review_enabled=True)
    ctx = AppContext(config=cfg, repo=repo, hub=hub)
    from chimera_patrol.ingest.camera import Camera

    ctx.cameras = [Camera(id="cam_a", name="东门", source_type="file", source="a.mp4"), Camera(id="cam_b", name="西门", source_type="file", source="b.mp4")]
    ctx.frame_slots = {"cam_a": FrameSlot(), "cam_b": FrameSlot()}
    app = create_app(ctx)
    client = TestClient(app)
    yield {"client": client, "repo": repo, "ctx": ctx, "base": base}
    client.close()
    repo.close()


def test_list_events_filters_and_pagination(env):
    client = env["client"]
    body = client.get("/api/events").json()
    assert body["total"] == 4
    assert body["limit"] == 50 and body["offset"] == 0
    # 排序：created_ms DESC
    assert [item["id"] for item in body["items"]] == [
        "evt_000000000004",
        "evt_000000000003",
        "evt_000000000002",
        "evt_000000000001",
    ]
    # 列表不含 trajectory
    assert "trajectory" not in body["items"][0]["cv"]

    cam = client.get("/api/events", params={"camera_id": "cam_a"}).json()
    assert cam["total"] == 2
    assert {item["camera_id"] for item in cam["items"]} == {"cam_a"}

    smoke = client.get("/api/events", params={"event_type": "smoke_fire"}).json()
    assert smoke["total"] == 1

    page = client.get("/api/events", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 4
    assert [item["id"] for item in page["items"]] == ["evt_000000000002", "evt_000000000001"]


def test_dismissed_events_remain_visible(env):
    client = env["client"]
    repo = env["repo"]
    repo.transition_status(
        "evt_000000000001",
        from_status="pending_review",
        to_status="dismissed",
        actor_type="system",
        actor="vlm_reviewer",
        reason=None,
        now_ms=NOW + 10,
    )
    body = client.get("/api/events", params={"status": "dismissed"}).json()
    assert [item["id"] for item in body["items"]] == ["evt_000000000001"]
    assert body["items"][0]["status"] == "dismissed"


def test_invalid_filter_returns_unified_error(env):
    resp = env["client"].get("/api/events", params={"event_type": "bogus"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_limit_out_of_range_returns_validation_error(env):
    resp = env["client"].get("/api/events", params={"limit": 500})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_detail_includes_trajectory_and_status_log(env):
    client = env["client"]
    body = client.get("/api/events/evt_000000000001").json()
    assert body["cv"]["trajectory"] == [[NOW + 1, 0.0, 0.0, 10.0, 10.0]]
    assert body["evidence"]["keyframe_annotated_url"] == "/media/cam_a/evt_000000000001/keyframe_annotated.jpg"
    assert body["evidence"]["clip_url"] is None
    assert body["vlm"] is None
    assert body["sop"] is None
    assert body["status_log"][0] == {
        "from_status": None,
        "to_status": "pending_review",
        "actor_type": "system",
        "actor": "cv_pipeline",
        "reason": None,
        "created_ms": NOW + 1,
    }


def test_unknown_event_returns_404_error_format(env):
    resp = env["client"].get("/api/events/evt_missing00000")
    assert resp.status_code == 404
    assert resp.json() == {"error": {"code": "not_found", "message": "event evt_missing00000 not found"}}


def test_override_success_and_error_codes(env):
    client = env["client"]
    url = "/api/events/evt_000000000002/status"

    ok = client.post(url, json={"to_status": "dismissed", "operator": "张三", "reason": "误报"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "dismissed"
    assert ok.json()["status_log"][-1]["actor"] == "张三"

    same = client.post(url, json={"to_status": "dismissed", "operator": "张三"})
    assert same.status_code == 409
    assert same.json()["error"]["code"] == "invalid_transition"

    pending_target = client.post(url, json={"to_status": "pending_review", "operator": "张三"})
    assert pending_target.status_code == 422
    assert pending_target.json()["error"]["code"] == "validation_error"

    too_long = client.post(url, json={"to_status": "confirmed", "operator": "x" * 33})
    assert too_long.status_code == 422

    reason_too_long = client.post(url, json={"to_status": "confirmed", "operator": "张三", "reason": "r" * 201})
    assert reason_too_long.status_code == 422

    missing = client.post("/api/events/evt_missing00000/status", json={"to_status": "confirmed", "operator": "张三"})
    assert missing.status_code == 404


def test_override_pending_review_is_conflict(env):
    client = env["client"]
    resp = client.post(
        "/api/events/evt_000000000003/status", json={"to_status": "confirmed", "operator": "张三"}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "invalid_transition"


def test_cameras_and_stats(env):
    client = env["client"]
    cams = client.get("/api/cameras").json()["items"]
    by_id = {c["id"]: c for c in cams}
    assert by_id["cam_a"]["status"] == "online"
    assert by_id["cam_a"]["latest_event"]["id"] == "evt_000000000002"
    assert by_id["cam_a"]["event_counts"] == {"total": 2, "pending_human": 1}
    assert "source" not in by_id["cam_a"]
    assert by_id["cam_a"]["stream_url"] == "/api/cameras/cam_a/stream.mjpg"

    stats = client.get("/api/stats").json()
    assert stats["totals"]["total_alarms"] == 4
    assert stats["totals"]["false_positive_reduction_rate"] == 0.0
    assert stats["filters"]["include_sop"] is False
    assert "generated_ms" in stats


def test_stats_export_csv_and_json(env):
    client = env["client"]
    csv_resp = client.get("/api/stats/export", params={"format": "csv"})
    assert csv_resp.status_code == 200
    assert "attachment" in csv_resp.headers["content-disposition"]
    text = csv_resp.content.decode("utf-8-sig")
    assert text.splitlines()[0] == "dimension,key,total_alarms,confirmed,dismissed,needs_human,review_failed,pending_review,pending_human,false_positive_reduction_rate"
    assert text.splitlines()[1].startswith("total,,4,")

    json_resp = client.get("/api/stats/export")
    assert json_resp.headers["content-type"].startswith("application/json")
    assert json.loads(json_resp.content)["totals"]["total_alarms"] == 4


def test_mjpeg_headers_and_first_part(env):
    client = env["client"]
    slot: FrameSlot = env["ctx"].frame_slots["cam_a"]
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    slot.put(image, NOW, 0)

    url = "/api/cameras/cam_a/stream.mjpg"
    with client.stream("GET", url, params={"fps": 30}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert "boundary=frame" in resp.headers["content-type"]
        first = next(resp.iter_bytes())
    assert first.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in first  # JPEG SOI


def test_mjpeg_unknown_camera_404(env):
    resp = env["client"].get("/api/cameras/cam_missing/stream.mjpg")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_media_path_traversal_rejected(env):
    client = env["client"]
    base: Path = env["base"]
    base.mkdir(parents=True, exist_ok=True)
    (base / "cam_a").mkdir(exist_ok=True)
    (base / "cam_a" / "ok.jpg").write_bytes(b"\xff\xd8ok")
    assert client.get("/media/cam_a/ok.jpg").status_code == 200
    assert client.get("/media/../chimera.db").status_code == 404
    assert client.get("/media/cam_a/..%2F..%2Fsecret").status_code == 404


def test_websocket_hello_ping_and_status_push(env):
    client = env["client"]
    with client.websocket_connect("/ws/events") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert hello["data"]["heartbeat_interval_s"] == 15
        assert hello["data"]["vlm_review_enabled"] is True
        assert isinstance(hello["seq"], int)

        ws.send_text(json.dumps({"type": "ping"}))
        pong = json.loads(ws.receive_text())
        assert pong["type"] == "pong"

        resp = client.post(
            "/api/events/evt_000000000002/status",
            json={"to_status": "confirmed", "operator": "张三"},
        )
        assert resp.status_code == 200

        msg = json.loads(ws.receive_text())
        assert msg["type"] == "event.updated"
        assert msg["data"]["id"] == "evt_000000000002"
        assert msg["data"]["status"] == "confirmed"
        assert "trajectory" not in msg["data"]["cv"]
        assert msg["seq"] > hello["seq"]


# ---------------------------------------------------------------- SOP 触发


def _sop_app(env, *, sop_cfg_camera_ids=("cam_a",)):
    ctx: AppContext = env["ctx"]
    cfg = ctx.config
    definition = SopDefinition(
        id="entry_prep",
        name="入场作业准备",
        camera_ids=list(sop_cfg_camera_ids),
        steps=[SopStep(id="wear_helmet", description="佩戴安全帽", criteria="头部戴帽", requires_sequence=False)],
    )
    ring = {"cam_a": RingBuffer(window_s=5, max_fps=10), "cam_b": RingBuffer(window_s=5, max_fps=10)}
    ctx.sop = SopChecker(
        definitions=[definition],
        repository=env["repo"],
        sop_repo=ctx.sop_repo,
        pipeline=_UnusedPipeline(),
        state=ctx.state,
        adapter=MockAdapter(default_verdict="uncertain"),
        ring_buffers=ring,
        sop_cfg=cfg.sop,
        evidence_cfg=cfg.evidence,
        timeout_s=5,
        semaphore=asyncio.Semaphore(1),
        notifier=None,
    )


def test_sop_trigger_errors(env):
    client = env["client"]
    _sop_app(env)

    unknown = client.post("/api/sop/runs", json={"sop_id": "nope", "camera_id": "cam_a", "operator": "张三"})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    not_applicable = client.post(
        "/api/sop/runs", json={"sop_id": "entry_prep", "camera_id": "cam_b", "operator": "张三"}
    )
    assert not_applicable.status_code == 422
    assert not_applicable.json()["error"]["code"] == "sop_not_applicable"

    env["repo"].set_camera_status("cam_a", "offline", NOW + 20)
    offline = client.post("/api/sop/runs", json={"sop_id": "entry_prep", "camera_id": "cam_a", "operator": "张三"})
    assert offline.status_code == 409
    assert offline.json()["error"]["code"] == "camera_offline"
    env["repo"].set_camera_status("cam_a", "online", NOW + 21)

    bad_operator = client.post("/api/sop/runs", json={"sop_id": "entry_prep", "camera_id": "cam_a", "operator": ""})
    assert bad_operator.status_code == 422
    assert bad_operator.json()["error"]["code"] == "validation_error"


def test_sop_trigger_accepts_and_finishes_without_frames(env):
    client = env["client"]
    _sop_app(env)
    resp = client.post("/api/sop/runs", json={"sop_id": "entry_prep", "camera_id": "cam_a", "operator": "张三"})
    assert resp.status_code == 202
    run = resp.json()
    assert run["status"] == "running"
    assert run["sample_count"] == 1
    assert run["steps"] == []

    # 窗口为 0 且无帧，后台任务应把运行置为 check_failed_retry（no_frames）。
    deadline = time.monotonic() + 5
    final = None
    while time.monotonic() < deadline:
        final = client.get(f"/api/sop/runs/{run['id']}").json()
        if final["status"] != "running":
            break
        time.sleep(0.05)
    assert final is not None
    assert final["status"] == "check_failed_retry"
    assert final["failure_reason"] == "no_frames"
    assert final["event_id"] is None
    assert final["steps"] == []

    retry = client.post(
        "/api/sop/runs",
        json={"sop_id": "entry_prep", "camera_id": "cam_a", "operator": "张三", "retry_of": run["id"]},
    )
    assert retry.status_code == 202
    assert retry.json()["retry_of"] == run["id"]

    missing = client.post(
        "/api/sop/runs",
        json={"sop_id": "entry_prep", "camera_id": "cam_a", "operator": "张三", "retry_of": "sop_missing00000"},
    )
    assert missing.status_code == 404
