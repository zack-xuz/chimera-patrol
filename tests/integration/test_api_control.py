"""暂停/恢复接口与只读配置端点的 API 测试。

- 暂停、恢复接口与 GET /api/cameras 的 paused 字段。
- 未知摄像头返回 404 统一错误格式。
- GET /api/config/effective 的结构，以及 API Key 脱敏：只返回是否已设置，绝不返回取值。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chimera_patrol.api.app import create_app
from chimera_patrol.api.context import AppContext
from chimera_patrol.api.ws import WsHub
from chimera_patrol.config import AppConfig, parse_config
from chimera_patrol.ingest.camera import Camera
from chimera_patrol.ingest.capture import FrameSlot
from chimera_patrol.store.repository import EventRepository

SECRET = "sk-test-SENTINEL-do-not-leak-9f3a"
NOW = 1_790_000_000_000


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VLM_API_KEY", SECRET)
    repo = EventRepository(tmp_path / "chimera.db")
    repo.upsert_camera("cam_a", name="东门", source_type="file", source="a.mp4", status="online", now_ms=NOW)
    repo.upsert_camera("cam_b", name="西门", source_type="file", source="b.mp4", status="online", now_ms=NOW)
    cfg = parse_config(
        {
            "evidence": {"base_dir": str(tmp_path / "evidence")},
            "vlm": {
                "openai_compatible": {
                    "base_url": "https://token.sensenova.cn/v1",
                    "model": "sensenova-6.8-flash",
                }
            },
            "rules": {"cooldown_s": 30, "smoke_fire": {"conf_threshold": 0.3}},
            "cameras": [
                {
                    "id": "cam_a",
                    "name": "东门",
                    "rois": [{"id": "gate", "type": "intrusion", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}],
                },
                {"id": "cam_b", "name": "西门"},
            ],
        },
        env={"VLM_API_KEY": SECRET},
    )
    ctx = AppContext(config=cfg, repo=repo, hub=WsHub(heartbeat_interval_s=15, vlm_review_enabled=True))
    ctx.cameras = [
        Camera(id="cam_a", name="东门", source_type="file", source="a.mp4", rois=cfg.cameras[0].rois),
        Camera(id="cam_b", name="西门", source_type="file", source="b.mp4"),
    ]
    ctx.frame_slots = {"cam_a": FrameSlot(), "cam_b": FrameSlot()}
    test_client = TestClient(create_app(ctx))
    yield test_client
    test_client.close()
    repo.close()


def _camera(client: TestClient, camera_id: str) -> dict:
    items = client.get("/api/cameras").json()["items"]
    return next(item for item in items if item["id"] == camera_id)


def test_pause_and_resume_update_paused_flag(client: TestClient) -> None:
    assert _camera(client, "cam_a")["paused"] is False

    paused = client.post("/api/cameras/cam_a/pause")
    assert paused.status_code == 200
    assert paused.json() == {"camera_id": "cam_a", "paused": True}
    assert _camera(client, "cam_a")["paused"] is True
    assert _camera(client, "cam_b")["paused"] is False

    resumed = client.post("/api/cameras/cam_a/resume")
    assert resumed.status_code == 200
    assert resumed.json() == {"camera_id": "cam_a", "paused": False}
    assert _camera(client, "cam_a")["paused"] is False


def test_pause_is_idempotent(client: TestClient) -> None:
    assert client.post("/api/cameras/cam_b/pause").status_code == 200
    assert client.post("/api/cameras/cam_b/pause").status_code == 200
    assert _camera(client, "cam_b")["paused"] is True


def test_pause_unknown_camera_returns_404(client: TestClient) -> None:
    for action in ("pause", "resume"):
        resp = client.post(f"/api/cameras/cam_missing/{action}")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "not_found"


def test_paused_state_does_not_change_camera_status(client: TestClient) -> None:
    client.post("/api/cameras/cam_a/pause")
    camera = _camera(client, "cam_a")
    assert camera["status"] == "online"  # 暂停只影响检测，采集状态不变


def test_effective_config_shape(client: TestClient) -> None:
    body = client.get("/api/config/effective").json()

    cam_a = next(c for c in body["cameras"] if c["id"] == "cam_a")
    assert cam_a["thresholds"] == {"person_conf": 0.4, "smoke_fire_conf": 0.3, "helmet_conf": 0.5}
    assert cam_a["cooldown_s"] == 30
    assert cam_a["rois"][0]["id"] == "gate"
    assert cam_a["rois"][0]["polygon"][0] == [0.1, 0.1]
    assert cam_a["paused"] is False

    assert body["rules"]["cooldown_s"] == 30
    assert body["rules"]["smoke_fire"]["conf_threshold"] == 0.3
    assert "reference_point" in body["rules"]

    vlm = body["vlm"]
    assert vlm["provider"] == "openai_compatible"
    assert vlm["review_enabled"] is True
    assert vlm["timeout_s"] == 20
    assert vlm["model"] == "sensenova-6.8-flash"
    assert vlm["base_url"] == "https://token.sensenova.cn/v1"


def test_effective_config_never_returns_api_key_value(client: TestClient) -> None:
    resp = client.get("/api/config/effective")
    assert resp.status_code == 200
    assert SECRET not in resp.text, "API key value leaked into the response"
    api_key = resp.json()["vlm"]["api_key"]
    assert api_key == {"env_var": "VLM_API_KEY", "configured": True}


def test_effective_config_reports_missing_key_without_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    repo = EventRepository(tmp_path / "chimera.db")
    cfg = AppConfig.model_validate({"vlm": {"review_enabled": False}, "evidence": {"base_dir": str(tmp_path)}})
    ctx = AppContext(config=cfg, repo=repo, hub=WsHub(heartbeat_interval_s=15, vlm_review_enabled=False))
    test_client = TestClient(create_app(ctx))
    try:
        body = test_client.get("/api/config/effective").json()
        assert body["vlm"]["api_key"] == {"env_var": "VLM_API_KEY", "configured": False}
    finally:
        test_client.close()
        repo.close()
