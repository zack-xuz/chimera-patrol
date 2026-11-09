"""Unit tests for chimera_patrol.mcp.vlm_server (task 9.1).

**Validates: Requirements 5.4, 13.4**
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from chimera_patrol.config import load_config
from chimera_patrol.mcp import vlm_server
from chimera_patrol.mcp.vlm_server import build_runtime_adapter, create_server, main

EXPECTED_TOOLS = {"review_event", "check_sop", "check_helmet_crop"}


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "vlm:\n"
        "  provider: mock\n"
        "  mock:\n"
        "    default_verdict: true_positive\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def server(mock_config_path: Path) -> Any:
    config = load_config(mock_config_path, env={})
    return create_server(config, build_runtime_adapter(config, env={}))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _structured(result: Any) -> dict[str, Any]:
    # FastMCP returns (content blocks, structured dict) for tools with a dict return type.
    content, structured = result
    assert content, "tool must return at least one content block"
    assert isinstance(structured, dict)
    return structured


def test_module_exposes_entry_points() -> None:
    assert callable(vlm_server.create_server)
    assert callable(vlm_server.main)
    assert vlm_server.SERVER_NAME == "chimera-vlm"


def test_three_tools_registered_with_expected_names(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


def test_review_event_with_mock_provider_returns_verdict_dict(server: Any) -> None:
    result = asyncio.run(
        server.call_tool(
            "review_event",
            {
                "event_id": "evt-1",
                "event_type": "intrusion",
                "camera_name": "cam-1",
                "cv_confidence": 0.8,
                "image_jpeg_base64": _b64(b"\xff\xd8fake-jpeg"),
            },
        )
    )
    payload = _structured(result)
    assert set(payload) == {"verdict", "confidence", "description", "suggestion"}
    assert payload["verdict"] == "true_positive"
    assert payload["confidence"] == pytest.approx(0.9)
    assert isinstance(payload["description"], str) and payload["description"]
    assert isinstance(payload["suggestion"], str)


def test_check_helmet_crop_with_mock_provider_returns_verdict_dict(server: Any) -> None:
    result = asyncio.run(
        server.call_tool(
            "check_helmet_crop",
            {"track_id": "trk-1", "camera_id": "cam-1", "image_jpeg_base64": _b64(b"\xff\xd8crop")},
        )
    )
    payload = _structured(result)
    assert payload == {"verdict": "uncertain", "confidence": 0.0}


def test_check_sop_with_mock_provider_returns_step_results(server: Any) -> None:
    result = asyncio.run(
        server.call_tool(
            "check_sop",
            {
                "run_id": "run-1",
                "sop_name": "demo",
                "steps": [
                    {"step_id": "s1", "description": "wear helmet", "criteria": "helmet visible"},
                    {"step_id": "s2-fail", "description": "close gate", "criteria": "gate closed"},
                ],
                "frames": [{"index": 0, "ts_ms": 0, "image_jpeg_base64": _b64(b"\xff\xd8f0")}],
            },
        )
    )
    steps = _structured(result)["steps"]
    assert [s["step_id"] for s in steps] == ["s1", "s2-fail"]
    assert steps[0]["status"] == "pass"
    assert steps[1]["status"] == "fail"
    assert steps[1]["evidence_frame"] == 0


def test_invalid_base64_is_rejected(server: Any) -> None:
    with pytest.raises(Exception, match="base64"):
        asyncio.run(
            server.call_tool(
                "check_helmet_crop",
                {"track_id": "trk-1", "camera_id": "cam-1", "image_jpeg_base64": "not base64!!"},
            )
        )


def test_main_returns_2_for_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(tmp_path / "missing.yaml")])
    assert code == 2
    assert "配置文件不存在" in capsys.readouterr().err
