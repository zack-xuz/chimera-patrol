"""Unit tests for chimera_patrol.vlm.mock.MockAdapter (task 8.3)."""

from __future__ import annotations

import asyncio

from chimera_patrol.vlm.mock import MockAdapter
from chimera_patrol.vlm.normalize import VLMVerdict
from chimera_patrol.vlm.protocol import (
    HelmetCropRequest,
    HelmetCropResult,
    ReviewRequest,
    SopCheckRequest,
    SopCheckResult,
    VLMAdapter,
)


def _review_req(camera_name: str = "cam-1", max_chars: int = 50) -> ReviewRequest:
    return ReviewRequest(
        event_id="evt_000000000001",
        event_type="intrusion",
        camera_name=camera_name,
        roi_name="zone-a",
        cv_confidence=0.8,
        image_jpeg=b"\xff\xd8\xff",
        description_max_chars=max_chars,
    )


def test_mock_satisfies_vlm_adapter_protocol() -> None:
    assert isinstance(MockAdapter("uncertain"), VLMAdapter)


def test_review_event_uses_default_verdict_without_camera_override() -> None:
    adapter = MockAdapter("false_positive", {})
    verdict = asyncio.run(adapter.review_event(_review_req("cam-x")))
    assert isinstance(verdict, VLMVerdict)
    assert verdict.verdict == "false_positive"
    assert verdict.confidence == 0.9


def test_by_camera_overrides_default_verdict() -> None:
    adapter = MockAdapter("uncertain", {"cam-1": "true_positive"})
    overridden = asyncio.run(adapter.review_event(_review_req("cam-1")))
    fallback = asyncio.run(adapter.review_event(_review_req("cam-2")))
    assert overridden.verdict == "true_positive"
    assert overridden.confidence == 0.9
    assert fallback.verdict == "uncertain"
    assert fallback.confidence == 0.5


def test_review_description_is_truncated_to_max_chars() -> None:
    adapter = MockAdapter("true_positive")
    verdict = asyncio.run(adapter.review_event(_review_req(max_chars=4)))
    assert len(verdict.description) == 4
    assert verdict.description == "模拟复核：确认为真实事件"[:4]
    assert verdict.suggestion != ""


def test_check_sop_preserves_step_order_and_marks_fail_hook() -> None:
    adapter = MockAdapter("uncertain")
    steps = [
        ("wear_helmet", "佩戴安全帽", "头盔可见"),
        ("fail_gloves", "佩戴手套", "手套可见"),
        ("wear_vest", "穿反光衣", "衣物可见"),
    ]
    frames = [(0, 1000, b"a"), (1, 2000, b"b")]
    result = asyncio.run(adapter.check_sop(SopCheckRequest("run_1", "entry", steps, frames)))
    assert isinstance(result, SopCheckResult)
    assert [s[0] for s in result.steps] == ["wear_helmet", "fail_gloves", "wear_vest"]
    assert [s[1] for s in result.steps] == ["pass", "fail", "pass"]
    assert result.steps[1][3] == 0
    assert result.steps[0][3] is None


def test_check_helmet_crop_defaults_to_uncertain_zero_confidence() -> None:
    adapter = MockAdapter("true_positive")
    result = asyncio.run(adapter.check_helmet_crop(HelmetCropRequest("trk_1", "cam-1", b"\xff\xd8")))
    assert result == HelmetCropResult(verdict="uncertain", confidence=0.0)
